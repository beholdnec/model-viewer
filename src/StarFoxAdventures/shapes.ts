import * as Viewer from '../viewer';
import { mat4 } from 'gl-matrix';
import { nArray } from '../util';
import { ColorTexture } from '../gfx/helpers/RenderTargetHelpers';
import { GfxDevice, GfxSampler, GfxVertexBufferDescriptor, GfxInputState, GfxInputLayout, GfxBuffer, GfxBufferUsage, GfxIndexBufferDescriptor, GfxBufferFrequencyHint } from '../gfx/platform/GfxPlatform';
import { GfxWrapMode, GfxMipFilterMode, GfxTexFilterMode } from '../gfx/platform/GfxPlatform';
import { GX_VtxDesc, GX_VtxAttrFmt, compileVtxLoaderMultiVat, LoadedVertexLayout, LoadedVertexData, GX_Array, VtxLoader, VertexAttributeInput, LoadedVertexPacket, compilePartialVtxLoader } from '../gx/gx_displaylist';
import { PacketParams, MaterialParams, GXMaterialHelperGfx, createInputLayout, ub_PacketParams, ub_PacketParamsBufferSize, fillPacketParamsData } from '../gx/gx_render';
import { GfxRenderInstManager, GfxRenderInst } from "../gfx/render/GfxRenderer";
import { makeStaticDataBuffer } from '../gfx/helpers/BufferHelpers';
import { GfxRenderCache } from '../gfx/render/GfxRenderCache';
import { Camera, computeViewMatrix } from '../Camera';
import ArrayBufferSlice from '../ArrayBufferSlice';
import { GXMaterial } from '../gx/gx_material';
import { colorNewFromRGBA } from '../Color';

import { SFAMaterial } from './materials';
import { SFAAnimationController } from './animation';
import { ModelViewState } from './models';
import { ViewState } from './util';
import { SceneRenderContext } from './render';

class MyShapeHelper {
    public inputState: GfxInputState;
    public inputLayout: GfxInputLayout;
    private zeroBuffer: GfxBuffer | null = null;
    private vertexBuffers: GfxBuffer[] = [];
    private indexBuffer: GfxBuffer;

    constructor(device: GfxDevice, cache: GfxRenderCache, public loadedVertexLayout: LoadedVertexLayout, public loadedVertexData: LoadedVertexData, dynamicVertices: boolean, dynamicIndices: boolean) {
        let usesZeroBuffer = false;
        for (let attrInput: VertexAttributeInput = 0; attrInput < VertexAttributeInput.COUNT; attrInput++) {
            const attrib = loadedVertexLayout.singleVertexInputLayouts.find((attrib) => attrib.attrInput === attrInput);
            if (attrib === undefined) {
                usesZeroBuffer = true;
                break;
            }
        }

        const buffers: GfxVertexBufferDescriptor[] = [];
        for (let i = 0; i < loadedVertexData.vertexBuffers.length; i++) {
            const vertexBuffer = device.createBuffer((loadedVertexData.vertexBuffers[i].byteLength + 3) / 4, GfxBufferUsage.VERTEX,
                dynamicVertices ? GfxBufferFrequencyHint.DYNAMIC : GfxBufferFrequencyHint.STATIC);
            this.vertexBuffers.push(vertexBuffer);

            buffers.push({
                buffer: vertexBuffer,
                byteOffset: 0,
            });
        }

        if (usesZeroBuffer) {
            // TODO(jstpierre): Move this to a global somewhere?
            this.zeroBuffer = makeStaticDataBuffer(device, GfxBufferUsage.VERTEX, new Uint8Array(16).buffer);
            buffers.push({
                buffer: this.zeroBuffer,
                byteOffset: 0,
            });
        }

        this.inputLayout = createInputLayout(device, cache, loadedVertexLayout);

        this.indexBuffer = device.createBuffer((loadedVertexData.indexData.byteLength + 3) / 4, GfxBufferUsage.INDEX,
            dynamicIndices ? GfxBufferFrequencyHint.DYNAMIC : GfxBufferFrequencyHint.STATIC);

        const indexBufferDesc: GfxIndexBufferDescriptor = {
            buffer: this.indexBuffer,
            byteOffset: 0,
        };
        this.inputState = device.createInputState(this.inputLayout, buffers, indexBufferDesc);

        this.uploadData(device, true, true);
    }

    public uploadData(device: GfxDevice, uploadVertices: boolean, uploadIndices: boolean) {
        const hostAccessPass = device.createHostAccessPass();

        if (uploadVertices) {
            for (let i = 0; i < this.loadedVertexData.vertexBuffers.length; i++) {
                hostAccessPass.uploadBufferData(this.vertexBuffers[i], 0, new Uint8Array(this.loadedVertexData.vertexBuffers[i]));
            }
        }

        if (uploadIndices) {
            hostAccessPass.uploadBufferData(this.indexBuffer, 0, new Uint8Array(this.loadedVertexData.indexData));
        }

        device.submitPass(hostAccessPass);
    }

    public setOnRenderInst(renderInst: GfxRenderInst, packet: LoadedVertexPacket | null = null): void {
        renderInst.allocateUniformBuffer(ub_PacketParams, ub_PacketParamsBufferSize);
        renderInst.setInputLayoutAndState(this.inputLayout, this.inputState);
        if (packet !== null)
            renderInst.drawIndexes(packet.indexCount, packet.indexOffset);
        else
            renderInst.drawIndexes(this.loadedVertexData.totalIndexCount);
    }

    public fillPacketParams(packetParams: PacketParams, renderInst: GfxRenderInst): void {
        let offs = renderInst.getUniformBufferOffset(ub_PacketParams);
        const d = renderInst.mapUniformBufferF32(ub_PacketParams);
        fillPacketParamsData(d, offs, packetParams);
    }

    public destroy(device: GfxDevice): void {
        device.destroyInputState(this.inputState);
        if (this.zeroBuffer !== null)
            device.destroyBuffer(this.zeroBuffer);
    }
}

interface ShapeConfig {
    matrix: mat4;
    boneMatrices: mat4[];
    camera: Camera;
}

// The vertices and polygons of a shape.
export class ShapeGeometry {
    private vtxLoader: VtxLoader;
    private loadedVertexData: LoadedVertexData;

    private shapeHelper: MyShapeHelper | null = null;
    private packetParams = new PacketParams();
    private scratchMtx = mat4.create();
    private verticesDirty = true;

    public pnMatrixMap: number[] = nArray(10, () => 0);
    private hasFineSkinning = false;
    public hasBetaFineSkinning = false;

    constructor(private vtxArrays: GX_Array[], vcd: GX_VtxDesc[], vat: GX_VtxAttrFmt[][], displayList: ArrayBufferSlice, private isDynamic: boolean) {
        this.vtxLoader = compileVtxLoaderMultiVat(vat, vcd);
        this.loadedVertexData = this.vtxLoader.parseDisplayList(displayList);
        this.vtxLoader = compilePartialVtxLoader(this.vtxLoader, this.loadedVertexData);
        this.reloadVertices();
    }

    public reloadVertices() {
        this.vtxLoader.loadVertexData(this.loadedVertexData, this.vtxArrays);
        this.verticesDirty = true;
    }

    public setPnMatrixMap(pnMatrixMap: number[], hasFineSkinning: boolean) {
        for (let i = 0; i < pnMatrixMap.length; i++) {
            this.pnMatrixMap[i] = pnMatrixMap[i];
        }
        this.hasFineSkinning = hasFineSkinning;
    }

    private computeModelView(dst: mat4, camera: Camera, modelMatrix: mat4): void {
        computeViewMatrix(dst, camera);
        mat4.mul(dst, dst, modelMatrix);
    }

    public setOnRenderInst(device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, config: ShapeConfig) {
        if (this.shapeHelper === null) {
            this.shapeHelper = new MyShapeHelper(device, renderInstManager.gfxRenderCache,
                this.vtxLoader.loadedVertexLayout, this.loadedVertexData, this.isDynamic, false);
            this.verticesDirty = false;
        } else if (this.verticesDirty) {
            this.shapeHelper.uploadData(device, true, false);
            this.verticesDirty = false;
        }

        this.shapeHelper.setOnRenderInst(renderInst);

        this.packetParams.clear();

        for (let i = 0; i < this.packetParams.u_PosMtx.length; i++) {
            // PNMTX 9 is used for fine-skinned vertices in models with fine-skinning enabled.
            if (this.hasFineSkinning && i === 9) {
                mat4.identity(this.scratchMtx);
            } else {
                mat4.copy(this.scratchMtx, config.boneMatrices[this.pnMatrixMap[i]]);
            }

            mat4.mul(this.scratchMtx, config.matrix, this.scratchMtx);

            this.computeModelView(this.packetParams.u_PosMtx[i], config.camera, this.scratchMtx);
        }

        this.shapeHelper.fillPacketParams(this.packetParams, renderInst);
    }
}

export interface ShapeMaterial {
    setOnRenderInst: (device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, modelMatrix: mat4, sceneCtx: SceneRenderContext, boneMatrices: mat4[], modelViewState: ModelViewState) => void;
}

function computeModelView(dst: mat4, camera: Camera, modelMatrix: mat4): void {
    computeViewMatrix(dst, camera);
    mat4.mul(dst, dst, modelMatrix);
}

export class CommonShapeMaterial implements ShapeMaterial {
    private material: SFAMaterial;
    private gxMaterial: GXMaterial | undefined;
    private materialHelper: GXMaterialHelperGfx;
    private materialParams = new MaterialParams();
    private furLayer: number = 0;
    private overrideIndMtx: (mat4 | undefined)[] = [];
    private viewState: ViewState | undefined;
    private scratchMtx = mat4.create();

    public constructor(private animController: SFAAnimationController) {
    }

    // Caution: Material is referenced, not copied.
    public setMaterial(material: SFAMaterial) {
        this.material = material;
        this.updateMaterialHelper();
    }

    private updateMaterialHelper() {
        if (this.gxMaterial !== this.material.getGXMaterial()) {
            this.gxMaterial = this.material.getGXMaterial();
            this.materialHelper = new GXMaterialHelperGfx(this.gxMaterial);
        }
    }

    public setFurLayer(layer: number) {
        this.furLayer = layer;
    }

    public setOverrideIndMtx(num: number, mtx?: mat4) {
        if (mtx !== undefined) {
            if (this.overrideIndMtx[num] !== undefined) {
                mat4.copy(this.overrideIndMtx[num]!, mtx);
            } else {
                this.overrideIndMtx[num] = mat4.clone(mtx);
            }
        } else {
            this.overrideIndMtx[num] = undefined;
        }
    }

    public setOnRenderInst(device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, modelMatrix: mat4, sceneCtx: SceneRenderContext, boneMatrices: mat4[], modelViewState: ModelViewState) {
        this.updateMaterialHelper();
        
        const materialOffs = this.materialHelper.allocateMaterialParams(renderInst);

        for (let i = 0; i < 8; i++) {
            const tex = this.material.getTexture(i);
            
            if (tex === undefined || tex === null) {
                this.materialParams.m_TextureMapping[i].reset();
            } else if (tex.kind === 'fb-color-downscaled-8x' || tex.kind === 'fb-color-downscaled-2x') {
                // TODO: Downscale to 1/8th size and apply filtering
                const sceneTex = sceneCtx.getSceneTexture();
                this.materialParams.m_TextureMapping[i].gfxTexture = sceneTex.gfxTexture;
                this.materialParams.m_TextureMapping[i].gfxSampler = sceneCtx.getSceneTextureSampler();
                this.materialParams.m_TextureMapping[i].width = sceneTex.width;
                this.materialParams.m_TextureMapping[i].height = sceneTex.height;
                this.materialParams.m_TextureMapping[i].lodBias = 0.0;
            } else if (tex.kind === 'texture') {
                this.materialParams.m_TextureMapping[i].gfxTexture = tex.texture.gfxTexture;
                this.materialParams.m_TextureMapping[i].gfxSampler = tex.texture.gfxSampler;
                this.materialParams.m_TextureMapping[i].width = tex.texture.width;
                this.materialParams.m_TextureMapping[i].height = tex.texture.height;
                this.materialParams.m_TextureMapping[i].lodBias = 0.0;
            } else if (tex.kind === 'fur-map') {
                const furMap = this.material.factory.getFurFactory().getLayer(this.furLayer);
                this.materialParams.m_TextureMapping[i].gfxTexture = furMap.gfxTexture;
                this.materialParams.m_TextureMapping[i].gfxSampler = furMap.gfxSampler;
                this.materialParams.m_TextureMapping[i].width = furMap.width;
                this.materialParams.m_TextureMapping[i].height = furMap.height;
                this.materialParams.m_TextureMapping[i].lodBias = 0.0;
            }
        }

        renderInst.setSamplerBindingsFromTextureMappings(this.materialParams.m_TextureMapping);

        if (this.viewState === undefined) {
            this.viewState = {
                viewerInput: sceneCtx.viewerInput,
                animController: this.animController,
                modelViewMtx: mat4.create(),
                invModelViewMtx: mat4.create(),
                outdoorAmbientColor: colorNewFromRGBA(1.0, 1.0, 1.0, 1.0),
            };
        }

        this.viewState.viewerInput = sceneCtx.viewerInput;
        this.viewState.outdoorAmbientColor = this.material.factory.getAmbientColor(modelViewState.ambienceNum);

        // mat4.mul(this.scratchMtx, boneMatrices[this.geom.pnMatrixMap[0]], modelMatrix);
        mat4.copy(this.scratchMtx, modelMatrix);
        computeModelView(this.viewState.modelViewMtx, sceneCtx.viewerInput.camera, this.scratchMtx);
        mat4.invert(this.viewState.invModelViewMtx, this.viewState.modelViewMtx);

        this.material.setupMaterialParams(this.materialParams, this.viewState);

        // XXX: test lighting
        // colorCopy(this.materialParams.u_Color[ColorKind.MAT0], White);
        // this.materialParams.u_Lights[0].Position = vec3.create(); // All light information is in view space. This centers the light on the camera.
        // this.materialParams.u_Lights[0].Color = colorNewFromRGBA(1.0, 1.0, 1.0, 1.0);
        // this.materialParams.u_Lights[0].CosAtten = vec3.fromValues(1.0, 0.0, 0.0);
        // this.materialParams.u_Lights[0].DistAtten = vec3.fromValues(1.0, 1/800, 1/800000);

        for (let i = 0; i < 3; i++) {
            if (this.overrideIndMtx[i] !== undefined) {
                mat4.copy(this.materialParams.u_IndTexMtx[i], this.overrideIndMtx[i]!);
            }
        }

        this.materialHelper.setOnRenderInst(device, renderInstManager.gfxRenderCache, renderInst);
        this.materialHelper.fillMaterialParamsDataOnInst(renderInst, materialOffs, this.materialParams);
    }
}

// The geometry and material of a shape.
export class Shape {
    public constructor(public geom: ShapeGeometry, public material: ShapeMaterial, public isDevGeometry: boolean) {
    }

    public reloadVertices() {
        this.geom.reloadVertices();
    }

    public setOnRenderInst(device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, modelMatrix: mat4, sceneCtx: SceneRenderContext, boneMatrices: mat4[], modelViewState: ModelViewState) {
        this.geom.setOnRenderInst(device, renderInstManager, renderInst, {
            matrix: modelMatrix,
            boneMatrices: boneMatrices,
            camera: sceneCtx.viewerInput.camera,
        });

        this.material.setOnRenderInst(device, renderInstManager, renderInst, modelMatrix, sceneCtx, boneMatrices, modelViewState);
    }

    public prepareToRender(device: GfxDevice, renderInstManager: GfxRenderInstManager, modelMatrix: mat4, sceneCtx: SceneRenderContext, boneMatrices: mat4[], modelViewState: ModelViewState) {
        const renderInst = renderInstManager.newRenderInst();
        this.setOnRenderInst(device, renderInstManager, renderInst, modelMatrix, sceneCtx, boneMatrices, modelViewState);
        renderInstManager.submitRenderInst(renderInst);
    }
}