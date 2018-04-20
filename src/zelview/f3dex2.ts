
import { mat4, vec3, vec4 } from 'gl-matrix';

import * as Render from './render';
import * as ZELVIEW0 from './zelview0';

import { CullMode, RenderState, RenderFlags, BlendMode } from '../render';
import Program from '../program';
import * as Viewer from '../viewer';

function extractBits(value: number, offset: number, bits: number) {
    return (value >> offset) & ((1 << bits) - 1);
}

// Zelda uses the F3DEX2 display list format. This implements
// a simple (and probably wrong!) HLE renderer for it.

type CmdFunc = (renderState: RenderState) => void;

const enum UCodeCommands {
    VTX = 0x01,
    TRI1 = 0x05,
    TRI2 = 0x06,
    GEOMETRYMODE = 0xD9,

    SETOTHERMODE_L = 0xE2,
    SETOTHERMODE_H = 0xE3,

    DL = 0xDE,
    ENDDL = 0xDF,

    MTX = 0xDA,
    POPMTX = 0xD8,

    TEXTURE = 0xD7,
    LOADTLUT = 0xF0,
    LOADBLOCK = 0xF3,
    LOADTILE = 0xF4,
    SETCIMG = 0xFF,
    SETZIMG = 0xFE,
    SETTIMG = 0xFD,
    SETTILESIZE = 0xF2,
    SETTILE = 0xF5,
    RDPLOADSYNC = 0xE6,
    RDPPIPESYNC = 0xE7,
    RDPTILESYNC = 0xE8,
    RDPFULLSYNC = 0xE9,
    FILLRECT = 0xF6,
    TEXRECT = 0xE4,
    TEXRECTFLIP = 0xE5,

    SETPRIMDEPTH = 0xEE,
    SETCONVERT = 0xEC,
    SETFILLCOLOR = 0xF7,
    SETFOGCOLOR = 0xF8,
    SETBLENDCOLOR = 0xF9,
    SETPRIMCOLOR = 0xFA,
    SETENVCOLOR = 0xFB,
    SETCOMBINE = 0xFC,
    SETKEYR = 0xEB,
    SETKEYGB = 0xEA,
}

const CCMUX = {
    COMBINED: 0,
    TEXEL0: 1,
    TEXEL1: 2,
    PRIMITIVE: 3,
    SHADE: 4,
    ENVIRONMENT: 5,
    CENTER: 6,
    SCALE: 6,
    COMBINED_ALPHA: 7,
    TEXEL0_ALPHA: 8,
    TEXEL1_ALPHA: 9,
    PRIMITIVE_ALPHA: 10,
    SHADE_ALPHA: 11,
    ENV_ALPHA: 12,
    LOD_FRACTION: 13,
    PRIM_LOD_FRAC: 14,
    NOISE: 7,
    K4: 7,
    K5: 15,
    _1: 6,
    _0: 31,
};

const ACMUX = {
    COMBINED: 0,
    TEXEL0: 1,
    TEXEL1: 2,
    PRIMITIVE: 3,
    SHADE: 4,
    ENVIRONMENT: 5,
    LOD_FRACTION: 0,
    PRIM_LOD_FRAC: 6,
    _1: 6,
    _0: 7,
};

const G_IM_FMT = {
    RGBA: 0,
    CI: 2,
    IA: 3,
    I: 4,
};

const G_IM_SIZ = {
    _4b: 0,
    _8b: 1,
    _16b: 2,
};

function imFmtSiz(fmt: number, siz: number) {
    return (fmt << 4) | siz;
}

interface TImgParams {
    fmt: number;
    siz: number;
    width: number;
    addr: number;
}

interface TileParams {
    fmt: number;
    siz: number;
    line: number;
    tmem: number;
    palette: number;
    cmt: number;
    cms: number;
    maskt: number;
    masks: number;
    shiftt: number;
    shifts: number;

    uls: number;
    ult: number;
    lrs: number;
    lrt: number;
}

interface LoadedTexture {
    glTextureId: WebGLTexture;
    width: number;
    height: number;
}

const TMEM_SIZE = 4096;
const TMEM_ADDR_MASK = 0xFFF;
const NUM_TILES = 8;

// A special data view for TMEM that masks addresses so no accesses are out of bounds.
class TmemDataView {
    private tmem: Uint8Array = new Uint8Array(TMEM_SIZE);
    private view: DataView;

    constructor(tmem: Uint8Array) {
        this.tmem = tmem;
        this.view = new DataView(this.tmem.buffer);
    }

    public getUint8(offset: number) {
        return this.view.getUint8(offset & TMEM_ADDR_MASK);
    }

    public getUint16(offset: number) {
        return (this.getUint8(offset) << 8) | this.getUint8(offset + 1);
    }
}

let loggedprogparams = 0;
class State {
    public gl: WebGL2RenderingContext;
    public programMap: {[hash: string]: Render.F3DEX2Program} = {};

    public cmds: CmdFunc[];
    public textures: Viewer.Texture[];
    public tmem: Uint8Array = new Uint8Array(TMEM_SIZE);

    public mtx: mat4;
    public mtxStack: mat4[];

    public vertexBuffer: Float32Array;
    public vertexData: number[];
    public vertexOffs: number;

    public primColor: vec4 = vec4.clone([1, 1, 1, 1]);
    public envColor: vec4 = vec4.clone([1, 1, 1, 1]);

    public geometryMode: number = 0;
    public combiners: Readonly<Render.Combiners>;
    public otherModeL: number = 0;
    public otherModeH: number = (CYCLETYPE._2CYCLE << OtherModeH.CYCLETYPE_SFT);
    public tex0TileNum: number = 0;
    public tex1TileNum: number = 1;

    public palettePixels: Uint8Array;
    public timgParams: Readonly<TImgParams>;
    public tileParams: Array<Readonly<TileParams>> = [];

    public rom: ZELVIEW0.ZELVIEW0;
    public banks: ZELVIEW0.RomBanks;

    constructor() {
        // Fill tile parameters with default values.
        for (let i = 0; i < NUM_TILES; i++) {
            this.tileParams[i] = Object.freeze({
                fmt: G_IM_FMT.RGBA,
                siz: G_IM_SIZ._16b,
                line: 0,
                tmem: 0,
                palette: 0,
                cmt: 0,
                cms: 0,
                maskt: 0,
                masks: 0,
                shiftt: 0,
                shifts: 0,
            
                uls: 0,
                ult: 0,
                lrs: 0,
                lrt: 0,
            });
        }
    }

    public lookupAddress(addr: number) {
        return this.rom.lookupAddress(this.banks, addr);
    }

    public getDLProgram(params: Render.F3DEX2ProgramParameters): Render.F3DEX2Program {
        const hash = Render.hashF3DEX2Params(params);
        if (!(hash in this.programMap)) {
            this.programMap[hash] = new Render.F3DEX2Program(params);
        }
        return this.programMap[hash];
    }

    public pushUseProgramCmds() {
        // Clone all relevant fields to prevent the closure from seeing different data than intended.
        // FIXME: is there a better way to do this?
        const envColor = vec4.clone(this.envColor);
        const primColor = vec4.clone(this.primColor);
        const geometryMode = this.geometryMode;
        const otherModeL = this.otherModeL;
        const otherModeH = this.otherModeH;

        const progParams: Render.F3DEX2ProgramParameters = Object.freeze({
            use2Cycle: (extractBits(otherModeH, OtherModeH.CYCLETYPE_SFT, OtherModeH.CYCLETYPE_LEN) == CYCLETYPE._2CYCLE),
            combiners: this.combiners,
        });

        let glTex0: WebGLTexture = null;
        let glTex0Dims = [1, 1];
        const loadTex0 = true; // TODO: Load texture only if necessary
        if (loadTex0) {
            let tileParams = this.tileParams[this.tex0TileNum];
            const loaded = loadTexture(this.gl, tileParams, new TmemDataView(this.tmem), tileParams.tmem * 8, this.palettePixels);
            glTex0 = loaded.glTextureId;
            glTex0Dims = [loaded.width, loaded.height];
        }

        let glTex1: WebGLTexture = null;
        let glTex1Dims = [1, 1];
        const loadTex1 = true; // TODO: Load texture only if necessary
        if (loadTex1) {
            let tileParams = this.tileParams[this.tex1TileNum];
            const loaded = loadTexture(this.gl, tileParams, new TmemDataView(this.tmem), tileParams.tmem * 8, this.palettePixels);
            glTex1 = loaded.glTextureId;
            glTex1Dims = [loaded.width, loaded.height];
        }
        
        if (loggedprogparams < 32) {
            console.log(`Program parameters: ${JSON.stringify(progParams, null, '\t')}`);
            loggedprogparams++;
        }

        // TODO: Don't call getDLProgram if state didn't change; it could be expensive.
        const prog = this.getDLProgram(progParams);

        let alphaTestMode: number;
        if (otherModeL & OtherModeL.FORCE_BL) {
            alphaTestMode = 0;
        } else {
            alphaTestMode = ((otherModeL & OtherModeL.CVG_X_ALPHA) ? 0x1 : 0 |
                                (otherModeL & OtherModeL.ALPHA_CVG_SEL) ? 0x2 : 0);
        }

        this.cmds.push((renderState: RenderState) => {
            const gl = renderState.gl;

            renderState.useProgram(prog);
            renderState.bindModelView();

            gl.uniform1i(prog.texture0Location, 0);
            gl.uniform1i(prog.texture1Location, 1);

            gl.uniform4fv(prog.envLocation, envColor);
            gl.uniform4fv(prog.primLocation, primColor);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, glTex0);
            gl.uniform2fv(prog.txsLocation[0], [1 / glTex0Dims[0], 1 / glTex0Dims[1]]);

            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, glTex1);
            gl.uniform2fv(prog.txsLocation[1], [1 / glTex1Dims[0], 1 / glTex1Dims[1]]);

            gl.activeTexture(gl.TEXTURE0);
            
            const lighting = !!(geometryMode & GeometryMode.LIGHTING);
            // When lighting is disabled, the vertex colors are passed to the rasterizer as the SHADE attribute.
            // When lighting is enabled, the vertex colors represent normals and SHADE is computed by the RSP.
            const useVertexColors = lighting ? 0 : 1;
            gl.uniform1i(prog.useVertexColorsLocation, useVertexColors);

            gl.uniform1i(prog.alphaTestLocation, alphaTestMode);
        });
    }
}

type TextureDestFormat = "i8" | "i8_a8" | "rgba8";

// 3 pos + 2 uv + 4 color/nrm
const VERTEX_SIZE = 9;
const VERTEX_BYTES = VERTEX_SIZE * Float32Array.BYTES_PER_ELEMENT;

function readVertex(state: State, which: number, addr: number) {
    const rom = state.rom;
    const offs = state.lookupAddress(addr);
    const posX = rom.view.getInt16(offs + 0, false);
    const posY = rom.view.getInt16(offs + 2, false);
    const posZ = rom.view.getInt16(offs + 4, false);

    const pos = vec3.clone([posX, posY, posZ]);
    vec3.transformMat4(pos, pos, state.mtx);

    const txU = rom.view.getInt16(offs + 8, false) * (1 / 32);
    const txV = rom.view.getInt16(offs + 10, false) * (1 / 32);

    const vtxArray = new Float32Array(state.vertexBuffer.buffer, which * VERTEX_BYTES, VERTEX_SIZE);
    vtxArray[0] = pos[0]; vtxArray[1] = pos[1]; vtxArray[2] = pos[2];
    vtxArray[3] = txU; vtxArray[4] = txV;

    vtxArray[5] = rom.view.getUint8(offs + 12) / 255;
    vtxArray[6] = rom.view.getUint8(offs + 13) / 255;
    vtxArray[7] = rom.view.getUint8(offs + 14) / 255;
    vtxArray[8] = rom.view.getUint8(offs + 15) / 255;
}

function cmd_VTX(state: State, w0: number, w1: number) {
    const N = (w0 >> 12) & 0xFF;
    const V0 = ((w0 >> 1) & 0x7F) - N;
    let addr = w1;

    for (let i = 0; i < N; i++) {
        const which = V0 + i;
        readVertex(state, which, addr);
        addr += 16;
    }
}

function flushDraw(state: State) {
    const gl = state.gl;

    const vtxBufSize = state.vertexData.length / VERTEX_SIZE;
    const vtxOffs = state.vertexOffs;
    const vtxCount = vtxBufSize - vtxOffs;
    state.vertexOffs = vtxBufSize;
    if (vtxCount === 0)
        return;

    state.pushUseProgramCmds();
    state.cmds.push((renderState: RenderState) => {
        const gl = renderState.gl;
        gl.drawArrays(gl.TRIANGLES, vtxOffs, vtxCount);
    });
}

function translateTRI(state: State, idxData: Uint8Array) {
    idxData.forEach((idx, i) => {
        const offs = idx * VERTEX_SIZE;
        for (let i = 0; i < VERTEX_SIZE; i++) {
            state.vertexData.push(state.vertexBuffer[offs + i]);
        }
    });
}

function tri(idxData: Uint8Array, offs: number, cmd: number) {
    idxData[offs + 0] = (cmd >> 17) & 0x7F;
    idxData[offs + 1] = (cmd >> 9) & 0x7F;
    idxData[offs + 2] = (cmd >> 1) & 0x7F;
}

function cmd_TRI1(state: State, w0: number, w1: number) {
    const idxData = new Uint8Array(3);
    tri(idxData, 0, w0);
    translateTRI(state, idxData);
}

function cmd_TRI2(state: State, w0: number, w1: number) {
    const idxData = new Uint8Array(6);
    tri(idxData, 0, w0); tri(idxData, 3, w1);
    translateTRI(state, idxData);
}

const GeometryMode = {
    CULL_FRONT: 0x0200,
    CULL_BACK: 0x0400,
    LIGHTING: 0x020000,
};

function cmd_GEOMETRYMODE(state: State, w0: number, w1: number) {
    flushDraw(state);

    state.geometryMode = state.geometryMode & ((~w0) & 0x00FFFFFF) | w1;
    const newMode = state.geometryMode;

    const renderFlags = new RenderFlags();

    const cullFront = newMode & GeometryMode.CULL_FRONT;
    const cullBack = newMode & GeometryMode.CULL_BACK;

    if (cullFront && cullBack)
        renderFlags.cullMode = CullMode.FRONT_AND_BACK;
    else if (cullFront)
        renderFlags.cullMode = CullMode.FRONT;
    else if (cullBack)
        renderFlags.cullMode = CullMode.BACK;
    else
        renderFlags.cullMode = CullMode.NONE;

    state.cmds.push((renderState: RenderState) => {
        renderState.useFlags(renderFlags);
    });
}

const OtherModeL = {
    Z_CMP: 0x0010,
    Z_UPD: 0x0020,
    ZMODE_DEC: 0x0C00,
    CVG_X_ALPHA: 0x1000,
    ALPHA_CVG_SEL: 0x2000,
    FORCE_BL: 0x4000,
};

let loggedsoml = 0;
function cmd_SETOTHERMODE_L(state: State, w0: number, w1: number) {
    flushDraw(state);

    const len = extractBits(w0, 0, 8) + 1;
    const sft = Math.max(0, 32 - extractBits(w0, 8, 8) - len);
    const mask = ((1 << len) - 1) << sft;

    if (loggedsoml < 32) {
        console.log(`SETOTHERMODE_L shift ${sft} len ${len} data 0x${w1.toString(16)}`);
        loggedsoml++;
    }

    state.otherModeL = (state.otherModeL & ~mask) | (w1 & mask);

    const renderFlags = new RenderFlags();
    const newMode = state.otherModeL;

    renderFlags.depthTest = !!(newMode & OtherModeL.Z_CMP);
    renderFlags.depthWrite = !!(newMode & OtherModeL.Z_UPD);

    let alphaTestMode: number;
    if (newMode & OtherModeL.FORCE_BL) {
        alphaTestMode = 0;
        renderFlags.blendMode = BlendMode.ADD;
    } else {
        alphaTestMode = ((newMode & OtherModeL.CVG_X_ALPHA) ? 0x1 : 0 |
                            (newMode & OtherModeL.ALPHA_CVG_SEL) ? 0x2 : 0);
        renderFlags.blendMode = BlendMode.NONE;
    }

    state.cmds.push((renderState: RenderState) => {
        const gl = renderState.gl;
        
        renderState.useFlags(renderFlags);

        if (newMode & OtherModeL.ZMODE_DEC) {
            gl.enable(gl.POLYGON_OFFSET_FILL);
            gl.polygonOffset(-0.5, -0.5);
        } else {
            gl.disable(gl.POLYGON_OFFSET_FILL);
        }
    });
}

const OtherModeH = {
    CYCLETYPE_SFT: 20,
    CYCLETYPE_LEN: 2,
};

const CYCLETYPE = {
    _1CYCLE: 0,
    _2CYCLE: 1,
    COPY: 2,
    FILL: 3,
}

let loggedsomh = 0;
function cmd_SETOTHERMODE_H(state: State, w0: number, w1: number) {
    flushDraw(state);

    const len = extractBits(w0, 0, 8) + 1;
    const sft = Math.max(0, 32 - extractBits(w0, 8, 8) - len);
    const mask = ((1 << len) - 1) << sft;

    if (loggedsomh < 32) {
        console.log(`SETOTHERMODE_H shift ${sft} len ${len} data 0x${w1.toString(16)}`);
        loggedsomh++;
    }

    state.otherModeH = (state.otherModeH & ~mask) | (w1 & mask);
}

function cmd_DL(state: State, w0: number, w1: number) {
    runDL(state, w1);
}

function cmd_MTX(state: State, w0: number, w1: number) {
    flushDraw(state);

    if (w1 & 0x80000000) state.mtx = state.mtxStack.pop();
    w1 &= ~0x80000000;

    state.mtxStack.push(state.mtx);
    state.mtx = mat4.clone(state.mtx);

    const rom = state.rom;
    let offs = state.lookupAddress(w1);

    const mtx = mat4.create();

    for (let x = 0; x < 4; x++) {
        for (let y = 0; y < 4; y++) {
            const mt1 = rom.view.getUint16(offs, false);
            const mt2 = rom.view.getUint16(offs + 32, false);
            mtx[(x * 4) + y] = ((mt1 << 16) | (mt2)) * (1 / 0x10000);
            offs += 2;
        }
    }

    mat4.multiply(state.mtx, state.mtx, mtx);
}

function cmd_POPMTX(state: State, w0: number, w1: number) {
    flushDraw(state);

    state.mtx = state.mtxStack.pop();
}

let loggedtexture = 0;
function cmd_TEXTURE(state: State, w0: number, w1: number) {
    flushDraw(state);

    const params = {
        scaleS: (extractBits(w1, 16, 16) + 1) / 65536.0, // FIXME: correct?
        scaleT: (extractBits(w1, 0, 16) + 1) / 65536.0, // FIXME: correct?
        level: extractBits(w0, 11, 3),
        tile: extractBits(w0, 8, 3),
        on: extractBits(w0, 1, 7),
    };

    if (loggedtexture < 32) {
        console.log(`TEXTURE ${JSON.stringify(params, null, '\t')}`);
        loggedtexture++;
    }

    state.tex0TileNum = params.tile;
    state.tex1TileNum = (params.tile + 1) & 0x7;
}

function r5g5b5a1(dst: Uint8Array, dstOffs: number, p: number) {
    let r, g, b, a;

    r = (p & 0xF800) >> 11;
    r = (r << (8 - 5)) | (r >> (10 - 8));

    g = (p & 0x07C0) >> 6;
    g = (g << (8 - 5)) | (g >> (10 - 8));

    b = (p & 0x003E) >> 1;
    b = (b << (8 - 5)) | (b >> (10 - 8));

    a = (p & 0x0001) ? 0xFF : 0x00;

    dst[dstOffs + 0] = r;
    dst[dstOffs + 1] = g;
    dst[dstOffs + 2] = b;
    dst[dstOffs + 3] = a;
}

let numCombinesLogged = 0;
function cmd_SETCOMBINE(state: State, w0: number, w1: number) {
    flushDraw(state);

    state.combiners = Object.freeze({
        colorCombiners: Object.freeze([
            Object.freeze({
                subA: extractBits(w0, 20, 4),
                subB: extractBits(w1, 28, 4),
                mul: extractBits(w0, 15, 5),
                add: extractBits(w1, 15, 3),
            }),
            Object.freeze({
                subA: extractBits(w0, 5, 4),
                subB: extractBits(w1, 24, 4),
                mul: extractBits(w0, 0, 5),
                add: extractBits(w1, 6, 3),
            }),
        ]),
        alphaCombiners: Object.freeze([
            Object.freeze({
                subA: extractBits(w0, 12, 3),
                subB: extractBits(w1, 12, 3),
                mul: extractBits(w0, 9, 3),
                add: extractBits(w1, 9, 3),
            }),
            Object.freeze({
                subA: extractBits(w1, 21, 3),
                subB: extractBits(w1, 3, 3),
                mul: extractBits(w1, 18, 3),
                add: extractBits(w1, 0, 3),
            }),
        ]),
    });

    if (numCombinesLogged < 16) {
        console.log(`SETCOMBINE ${JSON.stringify(state.combiners, null, '\t')}`);
        numCombinesLogged++;
    }
}

function cmd_SETENVCOLOR(state: State, w0: number, w1: number) {
    flushDraw(state);

    state.envColor = vec4.clone([
        extractBits(w1, 24, 8) / 255,
        extractBits(w1, 16, 8) / 255,
        extractBits(w1, 8, 8) / 255,
        extractBits(w1, 0, 8) / 255,
    ]);
}

function cmd_SETPRIMCOLOR(state: State, w0: number, w1: number) {
    flushDraw(state);

    state.primColor = vec4.clone([
        extractBits(w1, 24, 8) / 255,
        extractBits(w1, 16, 8) / 255,
        extractBits(w1, 8, 8) / 255,
        extractBits(w1, 0, 8) / 255,
    ]);
}

let loggedsettimg = 0;
function cmd_SETTIMG(state: State, w0: number, w1: number) {
    flushDraw(state);

    state.timgParams = Object.freeze({
        fmt: extractBits(w0, 21, 3),
        siz: extractBits(w0, 19, 2),
        width: extractBits(w0, 0, 12) + 1,
        addr: w1,
    });

    if (loggedsettimg < 32) {
        console.log(`SETTIMG ${JSON.stringify(state.timgParams, null, '\t')}`);
        loggedsettimg++;
    }
}

let loggedsettile = 0;
function cmd_SETTILE(state: State, w0: number, w1: number) {
    flushDraw(state);

    const tileIdx = extractBits(w1, 24, 3);
    let oldTile = Object.assign({}, state.tileParams[tileIdx]);
    state.tileParams[tileIdx] = Object.freeze(Object.assign(oldTile, {
        fmt: extractBits(w0, 21, 3),
        siz: extractBits(w0, 19, 2),
        line: extractBits(w0, 9, 9),
        tmem: extractBits(w0, 0, 9),
        tile: extractBits(w1, 24, 3),
        palette: extractBits(w1, 20, 4),
        cmt: extractBits(w1, 18, 2),
        cms: extractBits(w1, 8, 2),
        maskt: extractBits(w1, 14, 4),
        masks: extractBits(w1, 4, 4),
        shiftt: extractBits(w1, 10, 4),
        shifts: extractBits(w1, 0, 4),
        // Preserve uls, ult, lrs, lrt
    }));

    if (loggedsettile < 32) {
        console.log(`SETTILE ${JSON.stringify(state.tileParams[tileIdx], null, '\t')}`);
        loggedsettile++;
    }
}

function setTileSize(state: State, tileIdx: number, uls: number, ult: number, lrs: number, lrt: number) {
    let oldTile = Object.assign({}, state.tileParams[tileIdx]);
    state.tileParams[tileIdx] = Object.freeze(Object.assign(oldTile, {
        uls: uls,
        ult: ult,
        lrs: lrs,
        lrt: lrt,
    }));
}

let loggedsettilesize = 0;
function cmd_SETTILESIZE(state: State, w0: number, w1: number) {
    flushDraw(state);

    const params = Object.freeze({
        tile: extractBits(w1, 24, 3),
        uls: extractBits(w0, 12, 12),
        ult: extractBits(w0, 0, 12),
        lrs: extractBits(w1, 12, 12),
        lrt: extractBits(w1, 0, 12),
    });

    if (loggedsettilesize < 32) {
        console.log(`SETTILESIZE ${JSON.stringify(params, null, '\t')}`);
        loggedsettilesize++;
    }

    setTileSize(state, params.tile, params.uls, params.ult, params.lrs, params.lrt);
}

let loggedloadblock = 0;
function cmd_LOADBLOCK(state: State, w0: number, w1: number) {
    flushDraw(state);

    const params = Object.freeze({
        tile: extractBits(w1, 24, 3),
        uls: extractBits(w0, 12, 12),
        ult: extractBits(w0, 0, 12),
        lrs: extractBits(w1, 12, 12),
        dxt: extractBits(w1, 0, 12),
    });

    if (loggedloadblock < 32) {
        console.log(`LOADBLOCK ${JSON.stringify(params, null, '\t')}`);
        loggedloadblock++;
    }

    setTileSize(state, params.tile, params.uls, params.ult, params.lrs, params.dxt);

    const tileParams = state.tileParams[params.tile];
    // calculate bytes per line
    const bpl = state.timgParams.width << state.timgParams.siz >> 1;
    const srcOffs = state.lookupAddress(state.timgParams.addr + params.ult * bpl + (params.uls << state.timgParams.siz >> 1));
    const dstOffs = tileParams.tmem * 8;
    let numBytes = (params.lrs - params.uls + 1) << tileParams.siz >> 1;
    // Round up to next multiple of 8
    numBytes = (numBytes + 7) & ~7;
    for (let i = 0; i < numBytes; i++) {
        // TODO: blit
        state.tmem[(dstOffs + i) & TMEM_ADDR_MASK] = state.rom.view.getUint8(srcOffs + i);
    }
}

let loggedloadtile = 0;
function cmd_LOADTILE(state: State, w0: number, w1: number) {
    flushDraw(state);

    const params = Object.freeze({
        tile: extractBits(w1, 24, 3),
        uls: extractBits(w0, 12, 12),
        ult: extractBits(w0, 0, 12),
        lrs: extractBits(w1, 12, 12),
        lrt: extractBits(w1, 0, 12),
    });

    if (loggedloadtile < 32) {
        console.log(`LOADTILE ${JSON.stringify(params, null, '\t')}`);
        loggedloadtile++;
    }

    setTileSize(state, params.tile, params.uls, params.ult, params.lrs, params.lrt);
}

let loggedloadtlut = 0;
function cmd_LOADTLUT(state: State, w0: number, w1: number) {
    flushDraw(state);

    const params = Object.freeze({
        tile: extractBits(w1, 24, 3),
        uls: extractBits(w0, 12, 12),
        ult: extractBits(w0, 0, 12),
        lrs: extractBits(w1, 12, 12),
        lrt: extractBits(w1, 0, 12),
    });

    if (loggedloadtlut < 32) {
        console.log(`LOADTLUT ${JSON.stringify(params, null, '\t')}`);
        loggedloadtlut++;
    }

    setTileSize(state, params.tile, params.uls, params.ult, params.lrs, params.lrt);

    const rom = state.rom;

    // XXX: properly implement uls/ult/lrs/lrt
    const size = ((w1 & 0x00FFF000) >> 14) + 1;
    const dst = new Uint8Array(size * 4);

    // FIXME: which tile?
    let srcOffs = state.lookupAddress(state.timgParams.addr);
    let dstOffs = 0;

    for (let i = 0; i < size; i++) {
        const pixel = rom.view.getUint16(srcOffs, false);
        r5g5b5a1(dst, dstOffs, pixel);
        srcOffs += 2;
        dstOffs += 4;
    }

    state.palettePixels = dst;
}

interface ConvertResult {
    data: Uint8Array;
    glFormat: number;
}

function convert_CI4(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number, palette: Uint8Array): ConvertResult {
    if (!palette)
        return null;

    const nBytes = width * height * 4;
    const dst = new Uint8Array(nBytes);
    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x += 2) {
            // FIXME: ensure decoding works correctly when width is odd.
            const b = src.getUint8(lineOffs++);
            let idx;
    
            idx = ((b & 0xF0) >> 4) * 4;
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
    
            idx = (b & 0x0F) * 4;
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
        }
    }

    return {data: dst, glFormat: gl.RGBA};
}

function convert_I4(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number): ConvertResult {
    const nBytes = width * height * 2;
    const dst = new Uint8Array(nBytes);
    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x += 2) {
            // FIXME: ensure decoding works correctly when width is odd.
            const b = src.getUint8(lineOffs++);

            let p;
            p = (b & 0xF0) >> 4;
            p = p << 4 | p;
            dst[i++] = p;
            dst[i++] = p;
    
            p = (b & 0x0F);
            p = p << 4 | p;
            dst[i++] = p;
            dst[i++] = p;
        }
    }

    return {data: dst, glFormat: gl.LUMINANCE_ALPHA};
}

function convert_IA4(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number): ConvertResult {
    const nBytes = width * height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x++) {
            const b = src.getUint8(lineOffs++);
            let p; let pm;
    
            p = (b & 0xF0) >> 4;
            pm = p & 0x0E;
            dst[i++] = (pm << 4 | pm);
            dst[i++] = (p & 0x01) ? 0xFF : 0x00;
    
            p = (b & 0x0F);
            pm = p & 0x0E;
            dst[i++] = (pm << 4 | pm);
            dst[i++] = (p & 0x01) ? 0xFF : 0x00;
        }
    }

    return {data: dst, glFormat: gl.LUMINANCE_ALPHA};
}

function convert_CI8(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number, palette: Uint8Array): ConvertResult {
    if (!palette)
        return null;

    const nBytes = width * height * 4;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x++) {
            let idx = src.getUint8(lineOffs++) * 4;
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
        }
    }

    return {data: dst, glFormat: gl.RGBA};
}

function convert_I8(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number): ConvertResult {
    const nBytes = width * height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x++) {
            const p = src.getUint8(lineOffs++);
            dst[i++] = p;
            dst[i++] = p;
        }
    }

    return {data: dst, glFormat: gl.LUMINANCE_ALPHA};
}

function convert_IA8(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number): ConvertResult {
    const nBytes = width * height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x++) {
            const b = src.getUint8(lineOffs++);
            let p;
    
            p = (b & 0xF0) >> 4;
            p = p << 4 | p;
            dst[i++] = p;
    
            p = (b & 0x0F);
            p = p >> 4 | p;
            dst[i++] = p;
        }
    }

    return {data: dst, glFormat: gl.LUMINANCE_ALPHA};
}

function convert_RGBA16(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number): ConvertResult {
    const nBytes = width * height * 4;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x++) {
            const pixel = src.getUint16(lineOffs);
            r5g5b5a1(dst, i, pixel);
            i += 4;
            lineOffs += 2;
        }
    }

    return {data: dst, glFormat: gl.RGBA};
}

function convert_IA16(gl: WebGL2RenderingContext, src: TmemDataView, srcOffs: number, srcStride: number, width: number, height: number): ConvertResult {
    const nBytes = width * height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < height; y++) {
        let lineOffs = srcOffs + y * srcStride;
        for (let x = 0; x < width; x++) {
            dst[i++] = src.getUint8(lineOffs++);
            dst[i++] = src.getUint8(lineOffs++);
        }
    }

    return {data: dst, glFormat: gl.LUMINANCE_ALPHA};
}

// TODO: re-implement

// function textureToCanvas(texture: TextureTile): Viewer.Texture {
//     const canvas = document.createElement("canvas");
//     canvas.width = texture.width;
//     canvas.height = texture.height;

//     const ctx = canvas.getContext("2d");
//     const imgData = ctx.createImageData(canvas.width, canvas.height);

//     if (texture.dstFormat === "i8") {
//         for (let si = 0, di = 0; di < imgData.data.length; si++, di += 4) {
//             imgData.data[di + 0] = texture.pixels[si];
//             imgData.data[di + 1] = texture.pixels[si];
//             imgData.data[di + 2] = texture.pixels[si];
//             imgData.data[di + 3] = 255;
//         }
//     } else if (texture.dstFormat === "i8_a8") {
//         for (let si = 0, di = 0; di < imgData.data.length; si += 2, di += 4) {
//             imgData.data[di + 0] = texture.pixels[si];
//             imgData.data[di + 1] = texture.pixels[si];
//             imgData.data[di + 2] = texture.pixels[si];
//             imgData.data[di + 3] = texture.pixels[si + 1];
//         }
//     } else if (texture.dstFormat === "rgba8") {
//         imgData.data.set(texture.pixels);
//     }

//     try {
//         canvas.title = '0x' + texture.addr.toString(16) + '  ' + texture.format.toString(16) + '  ' + texture.dstFormat;
//     } catch (e) {
//         canvas.title = '(Malformed)'
//     }
//     ctx.putImageData(imgData, 0, 0);

//     const surfaces = [ canvas ];
//     return { name: canvas.title, surfaces };
// }

function loadTexture(gl: WebGL2RenderingContext, tileParams: TileParams, src: TmemDataView, srcOffs: number, palette: Uint8Array): LoadedTexture {
    const textureSize = calcTextureSize(tileParams);
    const srcStride = tileParams.line * 8;

    function convertTexturePixels(): ConvertResult {
        const fmtsiz = imFmtSiz(tileParams.fmt, tileParams.siz);
        switch (fmtsiz) {
        case imFmtSiz(G_IM_FMT.RGBA, G_IM_SIZ._16b):
            return convert_RGBA16(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height);
        case imFmtSiz(G_IM_FMT.CI, G_IM_SIZ._4b):
            return convert_CI4(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height, palette);
        case imFmtSiz(G_IM_FMT.CI, G_IM_SIZ._8b):
            return convert_CI8(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height, palette);
        case imFmtSiz(G_IM_FMT.IA, G_IM_SIZ._4b):
            return convert_IA4(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height);
        case imFmtSiz(G_IM_FMT.IA, G_IM_SIZ._8b):
            return convert_IA8(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height);
        case imFmtSiz(G_IM_FMT.IA, G_IM_SIZ._16b):
            return convert_IA16(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height);
        case imFmtSiz(G_IM_FMT.I, G_IM_SIZ._4b):
            return convert_I4(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height);
        case imFmtSiz(G_IM_FMT.I, G_IM_SIZ._8b):
            return convert_I8(gl, src, srcOffs, srcStride, textureSize.width, textureSize.height);
        // // 4-bit
        // case 0x40: return convert_CI4(gl, src, srcOffs, nTexels, palette);    // CI
        // case 0x60: return convert_IA4(gl, src, srcOffs, nTexels);    // IA
        // case 0x80: return convert_I4(gl, src, srcOffs, nTexels);     // I
        // // 8-bit
        // case 0x48: return convert_CI8(gl, src, srcOffs, nTexels, palette);    // CI
        // case 0x68: return convert_IA8(gl, src, srcOffs, nTexels);    // IA
        // case 0x88: return convert_I8(gl, src, srcOffs, nTexels);     // I
        // // 16-bit
        // case 0x10: return convert_RGBA16(gl, src, srcOffs, nTexels); // RGBA
        // case 0x70: return convert_IA16(gl, src, srcOffs, nTexels);   // IA
        default:
            console.error(`Unsupported tile format and size 0x${fmtsiz.toString(16)}`);
            return null;
        }
    }

    let converted = null;
    if (srcOffs !== null)
        converted = convertTexturePixels();

    function translateWrap(cm: number) {
        switch (cm) {
            case 1: return gl.MIRRORED_REPEAT;
            case 2: return gl.CLAMP_TO_EDGE;
            case 3: return gl.CLAMP_TO_EDGE;
            default: return gl.REPEAT;
        }
    }

    let texId = null;
    if (converted !== null) {
        texId = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texId);
        // Filters are set to NEAREST here because filtering is performed in the fragment shader.
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, translateWrap(tileParams.cms));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, translateWrap(tileParams.cmt));
        gl.texImage2D(gl.TEXTURE_2D, 0, converted.glFormat, textureSize.width, textureSize.height, 0, converted.glFormat, gl.UNSIGNED_BYTE, converted.data);
    } else {
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    return {glTextureId: texId, width: textureSize.width, height: textureSize.height};
}

interface TextureSize {
    width: number;
    height: number;
}

function calcTextureSize(tileParams: TileParams): TextureSize {
    let fmtsiz = imFmtSiz(tileParams.fmt, tileParams.siz);
    let maxTexel, lineShift;
    switch (fmtsiz) {
    case imFmtSiz(G_IM_FMT.RGBA, G_IM_SIZ._16b):
        maxTexel = 2048; lineShift = 2; break;
    case imFmtSiz(G_IM_FMT.CI, G_IM_SIZ._4b):
        maxTexel = 4096; lineShift = 4; break;
    case imFmtSiz(G_IM_FMT.CI, G_IM_SIZ._8b):
        maxTexel = 2048; lineShift = 3; break;
    case imFmtSiz(G_IM_FMT.IA, G_IM_SIZ._4b):
        maxTexel = 8196; lineShift = 4; break;
    case imFmtSiz(G_IM_FMT.IA, G_IM_SIZ._8b):
        maxTexel = 4096; lineShift = 3; break;
    case imFmtSiz(G_IM_FMT.IA, G_IM_SIZ._16b):
        maxTexel = 2048; lineShift = 2; break;
    case imFmtSiz(G_IM_FMT.I, G_IM_SIZ._4b):
        maxTexel = 8196; lineShift = 4; break;
    case imFmtSiz(G_IM_FMT.I, G_IM_SIZ._8b):
        maxTexel = 4096; lineShift = 3; break;
    // // 4-bit
    // case 0x00: maxTexel = 4096; lineShift = 4; break; // RGBA
    // case 0x40: maxTexel = 4096; lineShift = 4; break; // CI
    // case 0x60: maxTexel = 8196; lineShift = 4; break; // IA
    // case 0x80: maxTexel = 8196; lineShift = 4; break; // I
    // // 8-bit
    // case 0x08: maxTexel = 2048; lineShift = 3; break; // RGBA
    // case 0x48: maxTexel = 2048; lineShift = 3; break; // CI
    // case 0x68: maxTexel = 4096; lineShift = 3; break; // IA
    // case 0x88: maxTexel = 4096; lineShift = 3; break; // I
    // // 16-bit
    // case 0x10: maxTexel = 2048; lineShift = 2; break; // RGBA
    // case 0x50: maxTexel = 2048; lineShift = 0; break; // CI
    // case 0x70: maxTexel = 2048; lineShift = 2; break; // IA
    // case 0x90: maxTexel = 2048; lineShift = 0; break; // I
    // // 32-bit
    // case 0x18: maxTexel = 1024; lineShift = 2; break; // RGBA
    default:
        console.error(`Unsupported texture format and size 0x${fmtsiz.toString(16)}`);
        return {width: 8, height: 8}; // XXX: Return gibberish to get something on screen.
    }

    const lineW = tileParams.line << lineShift;
    const tileW = tileParams.lrs - tileParams.uls + 1;
    const tileH = tileParams.lrt - tileParams.ult + 1;

    const maskW = 1 << tileParams.maskt;
    const maskH = 1 << tileParams.maskt;

    let lineH;
    if (lineW > 0)
        lineH = Math.min(maxTexel / lineW, tileH);
    else
        lineH = 0;

    let width;
    if (tileParams.masks > 0 && (maskW * maskH) <= maxTexel)
        width = maskW;
    else if ((tileW * tileH) <= maxTexel)
        width = tileW;
    else
        width = lineW;

    let height;
    if (tileParams.maskt > 0 && (maskW * maskH) <= maxTexel)
        height = maskH;
    else if ((tileW * tileH) <= maxTexel)
        height = tileH;
    else
        height = lineH;

    return {width: width, height: height};
}

type CommandFunc = (state: State, w0: number, w1: number) => void;

const CommandDispatch: { [n: number]: CommandFunc } = {};
CommandDispatch[UCodeCommands.VTX] = cmd_VTX;
CommandDispatch[UCodeCommands.TRI1] = cmd_TRI1;
CommandDispatch[UCodeCommands.TRI2] = cmd_TRI2;
CommandDispatch[UCodeCommands.GEOMETRYMODE] = cmd_GEOMETRYMODE;
CommandDispatch[UCodeCommands.DL] = cmd_DL;
CommandDispatch[UCodeCommands.MTX] = cmd_MTX;
CommandDispatch[UCodeCommands.POPMTX] = cmd_POPMTX;
CommandDispatch[UCodeCommands.SETOTHERMODE_L] = cmd_SETOTHERMODE_L;
CommandDispatch[UCodeCommands.SETOTHERMODE_H] = cmd_SETOTHERMODE_H;
CommandDispatch[UCodeCommands.LOADBLOCK] = cmd_LOADBLOCK;
CommandDispatch[UCodeCommands.LOADTILE] = cmd_LOADTILE;
CommandDispatch[UCodeCommands.LOADTLUT] = cmd_LOADTLUT;
CommandDispatch[UCodeCommands.TEXTURE] = cmd_TEXTURE;
CommandDispatch[UCodeCommands.SETCOMBINE] = cmd_SETCOMBINE;
CommandDispatch[UCodeCommands.SETENVCOLOR] = cmd_SETENVCOLOR;
CommandDispatch[UCodeCommands.SETPRIMCOLOR] = cmd_SETPRIMCOLOR;
CommandDispatch[UCodeCommands.SETTIMG] = cmd_SETTIMG;
CommandDispatch[UCodeCommands.SETTILE] = cmd_SETTILE;
CommandDispatch[UCodeCommands.SETTILESIZE] = cmd_SETTILESIZE;

const F3DEX2 = {};

let warned = false;

function runDL(state: State, addr: number) {
    const rom = state.rom;
    let offs = state.lookupAddress(addr);
    if (offs === null)
        return;
    while (true) {
        const cmd0 = rom.view.getUint32(offs, false);
        const cmd1 = rom.view.getUint32(offs + 4, false);

        const cmdType = cmd0 >>> 24;
        if (cmdType === UCodeCommands.ENDDL)
            break;

        const func = CommandDispatch[cmdType];
        if (func)
            func(state, cmd0, cmd1);
        offs += 8;
    }

    flushDraw(state);
}

export class DL {
    constructor(public vao: WebGLVertexArrayObject, public cmds: CmdFunc[], public textures: Viewer.Texture[]) {
    }

    render(renderState: RenderState) {
        const gl = renderState.gl;
        gl.bindVertexArray(this.vao);
        this.cmds.forEach((cmd) => {
            cmd(renderState);
        })
        gl.bindVertexArray(null);
    }
}

export function readDL(gl: WebGL2RenderingContext, rom: ZELVIEW0.ZELVIEW0, banks: ZELVIEW0.RomBanks, startAddr: number): DL {
    const state = new State();

    state.gl = gl;
    state.cmds = [];
    state.textures = [];

    state.mtx = mat4.create();
    state.mtxStack = [state.mtx];

    state.vertexBuffer = new Float32Array(32 * VERTEX_SIZE);
    state.vertexData = [];
    state.vertexOffs = 0;

    state.rom = rom;
    state.banks = banks;

    runDL(state, startAddr);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vertBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(state.vertexData), gl.STATIC_DRAW);

    gl.vertexAttribPointer(Render.F3DEX2Program.a_Position, 3, gl.FLOAT, false, VERTEX_BYTES, 0);
    gl.vertexAttribPointer(Render.F3DEX2Program.a_UV, 2, gl.FLOAT, false, VERTEX_BYTES, 3 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribPointer(Render.F3DEX2Program.a_Shade, 4, gl.FLOAT, false, VERTEX_BYTES, 5 * Float32Array.BYTES_PER_ELEMENT);
    gl.enableVertexAttribArray(Render.F3DEX2Program.a_Position);
    gl.enableVertexAttribArray(Render.F3DEX2Program.a_UV);
    gl.enableVertexAttribArray(Render.F3DEX2Program.a_Shade);

    gl.bindVertexArray(null);

    return new DL(vao, state.cmds, state.textures);
}
