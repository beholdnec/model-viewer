import { mat4, vec4 } from 'gl-matrix';
import { GfxDevice, GfxVertexBufferDescriptor, GfxInputState, GfxInputLayout, GfxBuffer, GfxBufferUsage, GfxIndexBufferDescriptor, GfxBufferFrequencyHint } from '../gfx/platform/GfxPlatform';
import { GX_VtxDesc, GX_VtxAttrFmt, compileVtxLoaderMultiVat, LoadedVertexLayout, LoadedVertexData, GX_Array, VtxLoader, VertexAttributeInput, LoadedVertexPacket, VtxBlendInfo } from '../gx/gx_displaylist';
import { PacketParams, MaterialParams, GXMaterialHelperGfx, createInputLayout, ub_PacketParams, ub_PacketParamsBufferSize, fillPacketParamsData, ColorKind, VtxBlendParams, ub_VtxBlendParams, fillVtxBlendParamsData, ub_VtxBlendParamsBufferSize, GXShapeHelperGfx, loadedDataCoalescerComboGfx } from '../gx/gx_render';
import { GfxRenderInstManager, GfxRenderInst } from "../gfx/render/GfxRenderer";
import { makeStaticDataBuffer, GfxBufferCoalescerCombo } from '../gfx/helpers/BufferHelpers';
import { GfxRenderCache } from '../gfx/render/GfxRenderCache';
import { Camera, computeViewMatrix } from '../Camera';
import ArrayBufferSlice from '../ArrayBufferSlice';
import { GXMaterial } from '../gx/gx_material';
import { colorNewFromRGBA, colorCopy, White } from '../Color';

import { SFAMaterial } from './materials';
import { ModelRenderContext } from './models';
import { ViewState, computeModelView } from './util';

interface ShapeConfig {
    matrix: mat4;
    boneMatrices: mat4[];
    camera: Camera;
}

export interface VertexBlendingPiece {
    start: number;
    count: number;
    indices: vec4;
    weights: ArrayLike<vec4>;
}

interface BlendMtxSlot {
    slotNum: number;
    boneNum: number;
    invertBind: boolean;
}

// The vertices and polygons of a shape.
export class ShapeGeometry {
    private vtxLoader: VtxLoader;
    private loadedVertexData: LoadedVertexData;

    private shapeHelper: GXShapeHelperGfx | null = null;
    private bufferCoalescer: GfxBufferCoalescerCombo;
    private packetParams = new PacketParams();
    private vtxBlendParams = new VtxBlendParams();
    private scratchMtx = mat4.create();

    private pnMatrixMap: number[];

    private blendMtxSlots: BlendMtxSlot[] = [];

    constructor(private vtxArrays: GX_Array[], vcd: GX_VtxDesc[], vat: GX_VtxAttrFmt[][], displayList: ArrayBufferSlice, private useVtxBlends: boolean, pnMatrixMap: number[], private vertexBlendingPieces: VertexBlendingPiece[] = [], private invBindMatrices: mat4[] = []) {
        this.pnMatrixMap = [];
        for (let i = 0; i < pnMatrixMap.length; i++)
            this.pnMatrixMap.push(pnMatrixMap[i]);

        this.vtxLoader = compileVtxLoaderMultiVat(vat, vcd, useVtxBlends);
        this.loadedVertexData = this.vtxLoader.runVertices(this.vtxArrays, displayList, undefined, this.useVtxBlends ? this.vtxBlendInfo : undefined);
    }

    private findVertexBlendingPiece(posidx: number): VertexBlendingPiece | undefined {
        return this.vertexBlendingPieces.find((piece) => posidx >= piece.start && posidx < piece.start + piece.count);
    }

    private getMatrixPaletteIndexForBone(boneNum: number, invertBind: boolean) {
        const slot = this.blendMtxSlots.find((slot) => slot.boneNum === boneNum && slot.invertBind === invertBind);
        if (slot !== undefined) {
            return slot.slotNum;
        } else {
            const newSlot: BlendMtxSlot = {
                slotNum: this.blendMtxSlots.length,
                boneNum,
                invertBind,
            };
            this.blendMtxSlots.push(newSlot);
            return newSlot.slotNum;
        }
    }

    private vtxBlendInfo: VtxBlendInfo = {
        getBlendParams: (indices: Array<number>, weights: Array<number>, pnmtxidx: number = 0, posidx?: number) => {
            if (posidx !== undefined) {
                // Use position index to find blend params
                const piece = this.findVertexBlendingPiece(posidx);
                if (piece !== undefined) {
                    indices[0] = this.getMatrixPaletteIndexForBone(piece.indices[0], true);
                    indices[1] = this.getMatrixPaletteIndexForBone(piece.indices[1], true);
                    indices[2] = 0;
                    indices[3] = 0;
                    weights[0] = piece.weights[posidx - piece.start][0];
                    weights[1] = piece.weights[posidx - piece.start][1];
                    weights[2] = 0;
                    weights[3] = 0;
                    return;
                }
            }

            // Use PNMTXIDX to find blend params
            indices[0] = this.getMatrixPaletteIndexForBone(this.pnMatrixMap[pnmtxidx], false);
            indices[1] = 0;
            indices[2] = 0;
            indices[3] = 0;
            weights[0] = 1;
            weights[1] = 0;
            weights[2] = 0;
            weights[3] = 0;
        },
    }

    // Warning: Pieces are referenced, not copied.
    public setVertexBlendingPieces(pieces: VertexBlendingPiece[]) {
        this.vertexBlendingPieces = pieces;
    }

    private computeModelView(dst: mat4, camera: Camera, modelMatrix: mat4): void {
        computeViewMatrix(dst, camera);
        mat4.mul(dst, dst, modelMatrix);
    }

    public setOnRenderInst(device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, config: ShapeConfig) {
        if (this.shapeHelper === null) {
            this.bufferCoalescer = loadedDataCoalescerComboGfx(device, [this.loadedVertexData]);
            this.shapeHelper = new GXShapeHelperGfx(device, renderInstManager.gfxRenderCache,
                this.bufferCoalescer.coalescedBuffers[0].vertexBuffers,
                this.bufferCoalescer.coalescedBuffers[0].indexBuffer,
                this.vtxLoader.loadedVertexLayout, this.loadedVertexData);
        }

        this.shapeHelper.setOnRenderInst(renderInst);

        this.packetParams.clear();

        for (let i = 0; i < this.packetParams.u_PosMtx.length; i++) {
            // PNMTX 9 is used for fine-skinned vertices in models with fine-skinning enabled.
            if (this.useVtxBlends && i === 9) {
                mat4.identity(this.scratchMtx);
            } else {
                mat4.copy(this.scratchMtx, config.boneMatrices[this.pnMatrixMap[i]]);
            }

            mat4.mul(this.scratchMtx, config.matrix, this.scratchMtx);

            this.computeModelView(this.packetParams.u_PosMtx[i], config.camera, this.scratchMtx);
        }

        this.shapeHelper.fillPacketParams(this.packetParams, renderInst);

        this.computeModelView(this.vtxBlendParams.u_ModelView, config.camera, config.matrix);

        for (let i = 0; i < this.blendMtxSlots.length; i++) {
            const slot = this.blendMtxSlots[i];
            mat4.copy(this.vtxBlendParams.u_BlendMtx[i], config.boneMatrices[slot.boneNum]);
            if (slot.invertBind)
                mat4.mul(this.vtxBlendParams.u_BlendMtx[i], this.vtxBlendParams.u_BlendMtx[i], this.invBindMatrices[slot.boneNum]);
        }

        this.shapeHelper.fillVtxBlendParams(this.vtxBlendParams, renderInst);
    }
}

export interface ShapeMaterial {
    setOnRenderInst: (device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, modelMatrix: mat4, modelCtx: ModelRenderContext, boneMatrices: mat4[]) => void;
}

export class CommonShapeMaterial implements ShapeMaterial {
    private material: SFAMaterial;
    private gxMaterial: GXMaterial | undefined;
    private materialHelper: GXMaterialHelperGfx;
    private materialParams = new MaterialParams();
    private viewState: ViewState | undefined;
    private scratchMtx = mat4.create();

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

    public setOnRenderInst(device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, modelMatrix: mat4, modelCtx: ModelRenderContext) {
        this.updateMaterialHelper();
        
        const materialOffs = this.materialHelper.allocateMaterialParams(renderInst);

        if (this.viewState === undefined) {
            this.viewState = {
                sceneCtx: modelCtx,
                modelViewMtx: mat4.create(),
                invModelViewMtx: mat4.create(),
                outdoorAmbientColor: colorNewFromRGBA(1.0, 1.0, 1.0, 1.0),
                furLayer: modelCtx.furLayer,
            };
        }

        this.viewState.outdoorAmbientColor = this.material.factory.getAmbientColor(modelCtx.ambienceNum);

        // mat4.mul(this.scratchMtx, boneMatrices[this.geom.pnMatrixMap[0]], modelMatrix);
        mat4.copy(this.scratchMtx, modelMatrix);
        computeModelView(this.viewState.modelViewMtx, modelCtx.viewerInput.camera, this.scratchMtx);
        mat4.invert(this.viewState.invModelViewMtx, this.viewState.modelViewMtx);

        for (let i = 0; i < 8; i++) {
            const tex = this.material.getTexture(i);
            if (tex !== undefined) {
                tex.setOnTextureMapping(this.materialParams.m_TextureMapping[i], this.viewState);
            } else {
                this.materialParams.m_TextureMapping[i].reset();
            }
        }

        renderInst.setSamplerBindingsFromTextureMappings(this.materialParams.m_TextureMapping);

        this.material.setupMaterialParams(this.materialParams, this.viewState);

        // XXX: test lighting
        colorCopy(this.materialParams.u_Color[ColorKind.MAT0], White); // TODO
        modelCtx.setupLights(this.materialParams.u_Lights, modelCtx);

        for (let i = 0; i < 3; i++) {
            if (modelCtx.overrideIndMtx[i] !== undefined) {
                mat4.copy(this.materialParams.u_IndTexMtx[i], modelCtx.overrideIndMtx[i]!);
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

    public setOnRenderInst(device: GfxDevice, renderInstManager: GfxRenderInstManager, renderInst: GfxRenderInst, modelMatrix: mat4, modelCtx: ModelRenderContext, boneMatrices: mat4[]) {
        this.geom.setOnRenderInst(device, renderInstManager, renderInst, {
            matrix: modelMatrix,
            boneMatrices: boneMatrices,
            camera: modelCtx.viewerInput.camera,
        });

        this.material.setOnRenderInst(device, renderInstManager, renderInst, modelMatrix, modelCtx, boneMatrices);
    }

    public prepareToRender(device: GfxDevice, renderInstManager: GfxRenderInstManager, modelMatrix: mat4, modelCtx: ModelRenderContext, boneMatrices: mat4[]) {
        const renderInst = renderInstManager.newRenderInst();
        this.setOnRenderInst(device, renderInstManager, renderInst, modelMatrix, modelCtx, boneMatrices);
        renderInstManager.submitRenderInst(renderInst);
    }
}