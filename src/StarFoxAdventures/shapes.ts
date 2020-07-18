import { mat4, vec3, vec4 } from 'gl-matrix';
import { GfxDevice, GfxFormat } from '../gfx/platform/GfxPlatform';
import * as GX from '../gx/gx_enum';
import { GX_VtxDesc, GX_VtxAttrFmt, compileVtxLoaderMultiVat, LoadedVertexData, GX_Array, VtxLoader, VtxLoaderCustomizer, VertexAttributeInput, SingleVertexInputLayout, CustomVtxInput } from '../gx/gx_displaylist';
import { PacketParams, MaterialParams, GXMaterialHelperGfx, ColorKind, VtxBlendParams, GXShapeHelperGfx, loadedDataCoalescerComboGfx } from '../gx/gx_render';
import { GfxRenderInstManager, GfxRenderInst } from "../gfx/render/GfxRenderer";
import { GfxBufferCoalescerCombo } from '../gfx/helpers/BufferHelpers';
import { Camera, computeViewMatrix } from '../Camera';
import ArrayBufferSlice from '../ArrayBufferSlice';
import { GXMaterial } from '../gx/gx_material';
import { colorNewFromRGBA, colorCopy, White } from '../Color';

import { SFAMaterial } from './materials';
import { ModelRenderContext } from './models';
import { ViewState, computeModelView } from './util';
import { getSystemEndianness, Endianness } from '../endian';

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

class VtxBlendingCustomizer extends VtxLoaderCustomizer {
    public equals(other: VtxLoaderCustomizer) {
        return other instanceof VtxBlendingCustomizer;
    }

    public getCustomVtxInputs(): CustomVtxInput[] {
        return [
            // Indices
            {attrInput: VertexAttributeInput.COUNT + 0, format: GfxFormat.F32_RGBA}, // TODO: use integer format?
            // Weights
            {attrInput: VertexAttributeInput.COUNT + 1, format: GfxFormat.F32_RGBA},
        ];
    }

    public compilePostLoader(S: string, customInputs: SingleVertexInputLayout[]): string {
        function compileWriteOneComponentF32(offs: number, value: string): string {
            const littleEndian = (getSystemEndianness() === Endianness.LITTLE_ENDIAN);
            const dstOffs = `dstVertexDataOffs + ${offs}`;
            return `dstVertexDataView.setFloat32(${dstOffs}, ${value}, ${littleEndian})`;
        }

        function findCustomInput(attrInput: VertexAttributeInput) {
            return customInputs.find((input) => input.attrInput === attrInput);
        }

        let dstOffs = findCustomInput(VertexAttributeInput.COUNT + 0)!.bufferOffset;
        S += `
    const blendIndices = [0, 0, 0, 0];
    const blendWeights = [1, 0, 0, 0];
    customInfo.getBlendParams(blendIndices, blendWeights, pnmtxidx, idx${GX.Attr.POS});

    // BLENDINDICES
    ${compileWriteOneComponentF32(dstOffs + 0, `blendIndices[0]`)};
    ${compileWriteOneComponentF32(dstOffs + 4, `blendIndices[1]`)};
    ${compileWriteOneComponentF32(dstOffs + 8, `blendIndices[2]`)};
    ${compileWriteOneComponentF32(dstOffs + 12, `blendIndices[3]`)};
`;

        dstOffs = findCustomInput(VertexAttributeInput.COUNT + 1)!.bufferOffset;
        S += `
    // BLENDWEIGHTS
    ${compileWriteOneComponentF32(dstOffs + 0, `blendWeights[0]`)};
    ${compileWriteOneComponentF32(dstOffs + 4, `blendWeights[1]`)};
    ${compileWriteOneComponentF32(dstOffs + 8, `blendWeights[2]`)};
    ${compileWriteOneComponentF32(dstOffs + 12, `blendWeights[3]`)};
`;
        return S;
    }
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

    private vtxBlendingCustomizer?: VtxBlendingCustomizer;
    private blendMtxSlots: BlendMtxSlot[] = [];

    constructor(private vtxArrays: GX_Array[], vcd: GX_VtxDesc[], vat: GX_VtxAttrFmt[][], displayList: ArrayBufferSlice, private useVtxBlends: boolean, pnMatrixMap: number[], private vertexBlendingPieces: VertexBlendingPiece[] = [], private invBindTranslations: vec3[] = []) {
        this.pnMatrixMap = [];
        for (let i = 0; i < pnMatrixMap.length; i++)
            this.pnMatrixMap.push(pnMatrixMap[i]);

        if (this.useVtxBlends)
            this.vtxBlendingCustomizer = new VtxBlendingCustomizer();
        this.vtxLoader = compileVtxLoaderMultiVat(vat, vcd, this.vtxBlendingCustomizer);
        this.loadedVertexData = this.vtxLoader.runVertices(this.vtxArrays, displayList, undefined, this.vtxBlendInfo);
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

    private vtxBlendInfo = {
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
                mat4.translate(this.vtxBlendParams.u_BlendMtx[i], this.vtxBlendParams.u_BlendMtx[i], this.invBindTranslations[slot.boneNum]);
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

        computeModelView(this.viewState.modelViewMtx, modelCtx.viewerInput.camera, modelMatrix);
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