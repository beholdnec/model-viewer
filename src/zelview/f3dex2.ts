
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
}

const TMEM_SIZE = 4096;

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
    public textureTiles: Array<TextureTile> = [];

    public rom: ZELVIEW0.ZELVIEW0;
    public banks: ZELVIEW0.RomBanks;

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

        let glTex0: WebGLTexture = null;
        let glTex0Dims = [1, 1];
        if (this.textureTiles[this.tex0TileNum]) {
            let tile = this.textureTiles[this.tex0TileNum];
            loadTexture(this.gl, tile, new DataView(this.tmem.buffer), tile.tmem, this.palettePixels);
            glTex0 = tile.glTextureId;
            glTex0Dims = [tile.width, tile.height];
        }

        let glTex1: WebGLTexture = null;
        let glTex1Dims = [1, 1];
        if (this.textureTiles[this.tex1TileNum]) {
            let tile = this.textureTiles[this.tex1TileNum];
            loadTexture(this.gl, tile, new DataView(this.tmem.buffer), tile.tmem, this.palettePixels);
            glTex1 = tile.glTextureId;
            glTex1Dims = [tile.width, tile.height];
        }

        const progParams: Render.F3DEX2ProgramParameters = Object.freeze({
            use2Cycle: (extractBits(otherModeH, OtherModeH.CYCLETYPE_SFT, OtherModeH.CYCLETYPE_LEN) == CYCLETYPE._2CYCLE),
            combiners: this.combiners,
        });
        
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
            
            const lighting = geometryMode & GeometryMode.LIGHTING;
            //const useVertexColors = lighting ? 0 : 1;
            // TODO: implement lighting
            const useVertexColors = 1;
            gl.uniform1i(prog.useVertexColorsLocation, useVertexColors);

            gl.uniform1i(prog.alphaTestLocation, alphaTestMode);
        });
    }
}

type TextureDestFormat = "i8" | "i8_a8" | "rgba8";

interface TextureTile {
    width: number;
    height: number;
    pixels: Uint8Array;
    addr: number;
    format: number;
    dstFormat: TextureDestFormat;

    // XXX(jstpierre): Move somewhere else?
    glTextureId: WebGLTexture;

    tmem: number;

    // Internal size data.
    lrs: number; lrt: number;
    uls: number; ult: number;
    maskS: number; maskT: number; lineSize: number;

    // wrap modes
    cms: number; cmt: number;
}

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

function flushTexture(state: State) {
    for (let i = 0; i < state.textureTiles.length; i++) {
        if (state.textureTiles[i])
            loadTile(state, state.textureTiles[i]);
    }
}

function cmd_TRI1(state: State, w0: number, w1: number) {
    flushTexture(state);
    const idxData = new Uint8Array(3);
    tri(idxData, 0, w0);
    translateTRI(state, idxData);
}

function cmd_TRI2(state: State, w0: number, w1: number) {
    flushTexture(state);
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
    state.tileParams[tileIdx] = Object.freeze({
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
    });

    if (loggedsettile < 32) {
        console.log(`SETTILE ${JSON.stringify(state.tileParams[tileIdx], null, '\t')}`);
        loggedsettile++;
    }

    // TODO: remove and rework
    state.textureTiles[tileIdx] = {
        format: (w0 >> 16) & 0xFF,
        cms: (w1 >> 8) & 0x3,
        cmt: (w1 >> 18) & 0x3,
        // tmem: w0 & 0x1FF,
        lineSize: (w0 >> 9) & 0x1FF,
        // palette: (w1 >> 20) & 0xF,
        // shiftS: w1 & 0xF,
        // shiftT: (w1 >> 10) & 0xF,
        maskS: (w1 >> 4) & 0xF,
        maskT: (w1 >> 14) & 0xF,

        tmem: state.tileParams[tileIdx].tmem,

        width: 0, height: 0, dstFormat: null,
        pixels: null, addr: 0, glTextureId: null,
        uls: 0, ult: 0, lrs: 0, lrt: 0,
    };
}

function setTileSize(state: State, tileIdx: number, uls: number, ult: number, lrs: number, lrt: number) {
    const tile = state.textureTiles[tileIdx];
    tile.uls = uls;
    tile.ult = ult;
    tile.lrs = lrs;
    tile.lrt = lrt;
    calcTextureSize(tile);
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
    const dstOffs = tileParams.tmem;
    let numBytes = (params.lrs - params.uls + 1) << tileParams.siz >> 1;
    if ((numBytes & 7) != 0) {
        // Round up to next multiple of 8
        numBytes = (numBytes & ~7) + 8;
    }
    for (let i = 0; i < numBytes; i++) {
        // TODO: blit
        state.tmem[dstOffs + i] = state.rom.view.getUint8(srcOffs + i);
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

function tileCacheKey(state: State, tile: TextureTile) {
    // XXX: Do we need more than this?
    const srcOffs = state.lookupAddress(tile.addr);
    return srcOffs;
}

// XXX: This is global to cut down on resources between DLs.
const tileCache = new Map<number, TextureTile>();
function loadTile(state: State, texture: TextureTile) {
    if (texture.glTextureId)
        return;

    const key = tileCacheKey(state, texture);
    const otherTile = tileCache.get(key);
    if (!otherTile) {
        const srcOffs = state.lookupAddress(texture.addr);
        loadTexture(state.gl, texture, state.rom.view, srcOffs, state.palettePixels);
        state.textures.push(textureToCanvas(texture));
        tileCache.set(key, texture);
    } else if (texture !== otherTile) {
        texture.glTextureId = otherTile.glTextureId;
    }
}

function convert_CI4(texture: TextureTile, src: DataView, srcOffs: number, palette: Uint8Array) {
    if (!palette)
        return;

    const nBytes = texture.width * texture.height * 4;
    const dst = new Uint8Array(nBytes);
    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x += 2) {
            const b = src.getUint8(srcOffs++);
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

    texture.pixels = dst;
}

function convert_I4(texture: TextureTile, src: DataView, srcOffs: number) {
    const nBytes = texture.width * texture.height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x += 2) {
            const b = src.getUint8(srcOffs++);

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

    texture.pixels = dst;
}

function convert_IA4(texture: TextureTile, src: DataView, srcOffs: number) {
    const nBytes = texture.width * texture.height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x += 2) {
            const b = src.getUint8(srcOffs++);
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

    texture.pixels = dst;
}

function convert_CI8(texture: TextureTile, src: DataView, srcOffs: number, palette: Uint8Array) {
    if (!palette)
        return;

    const nBytes = texture.width * texture.height * 4;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x++) {
            let idx = src.getUint8(srcOffs) * 4;
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            dst[i++] = palette[idx++];
            srcOffs++;
        }
    }

    texture.pixels = dst;
}

function convert_I8(texture: TextureTile, src: DataView, srcOffs: number) {
    const nBytes = texture.width * texture.height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x++) {
            const p = src.getUint8(srcOffs++);
            dst[i++] = p;
            dst[i++] = p;
        }
    }

    texture.pixels = dst;
}

function convert_IA8(texture: TextureTile, src: DataView, srcOffs: number) {
    const nBytes = texture.width * texture.height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x++) {
            const b = src.getUint8(srcOffs++);
            let p;

            p = (b & 0xF0) >> 4;
            p = p << 4 | p;
            dst[i++] = p;

            p = (b & 0x0F);
            p = p >> 4 | p;
            dst[i++] = p;
        }
    }

    texture.pixels = dst;
}

function convert_RGBA16(texture: TextureTile, src: DataView, srcOffs: number) {
    const nBytes = texture.width * texture.height * 4;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x++) {
            const pixel = src.getUint16(srcOffs, false);
            r5g5b5a1(dst, i, pixel);
            i += 4;
            srcOffs += 2;
        }
    }

    texture.pixels = dst;
}

function convert_IA16(texture: TextureTile, src: DataView, srcOffs: number) {
    const nBytes = texture.width * texture.height * 2;
    const dst = new Uint8Array(nBytes);

    let i = 0;
    for (let y = 0; y < texture.height; y++) {
        for (let x = 0; x < texture.width; x++) {
            dst[i++] = src.getUint8(srcOffs++);
            dst[i++] = src.getUint8(srcOffs++);
        }
    }

    texture.pixels = dst;
}

function textureToCanvas(texture: TextureTile): Viewer.Texture {
    const canvas = document.createElement("canvas");
    canvas.width = texture.width;
    canvas.height = texture.height;

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(canvas.width, canvas.height);

    if (texture.dstFormat === "i8") {
        for (let si = 0, di = 0; di < imgData.data.length; si++, di += 4) {
            imgData.data[di + 0] = texture.pixels[si];
            imgData.data[di + 1] = texture.pixels[si];
            imgData.data[di + 2] = texture.pixels[si];
            imgData.data[di + 3] = 255;
        }
    } else if (texture.dstFormat === "i8_a8") {
        for (let si = 0, di = 0; di < imgData.data.length; si += 2, di += 4) {
            imgData.data[di + 0] = texture.pixels[si];
            imgData.data[di + 1] = texture.pixels[si];
            imgData.data[di + 2] = texture.pixels[si];
            imgData.data[di + 3] = texture.pixels[si + 1];
        }
    } else if (texture.dstFormat === "rgba8") {
        imgData.data.set(texture.pixels);
    }

    try {
        canvas.title = '0x' + texture.addr.toString(16) + '  ' + texture.format.toString(16) + '  ' + texture.dstFormat;
    } catch (e) {
        canvas.title = '(Malformed)'
    }
    ctx.putImageData(imgData, 0, 0);

    const surfaces = [ canvas ];
    return { name: canvas.title, surfaces };
}

function loadTexture(gl: WebGL2RenderingContext, texture: TextureTile, src: DataView, srcOffs: number, palette: Uint8Array) {
    function convertTexturePixels() {
        switch (texture.format) {
        // 4-bit
        case 0x40: return convert_CI4(texture, src, srcOffs, palette);    // CI
        case 0x60: return convert_IA4(texture, src, srcOffs);    // IA
        case 0x80: return convert_I4(texture, src, srcOffs);     // I
        // 8-bit
        case 0x48: return convert_CI8(texture, src, srcOffs, palette);    // CI
        case 0x68: return convert_IA8(texture, src, srcOffs);    // IA
        case 0x88: return convert_I8(texture, src, srcOffs);     // I
        // 16-bit
        case 0x10: return convert_RGBA16(texture, src, srcOffs); // RGBA
        case 0x70: return convert_IA16(texture, src, srcOffs);   // IA
        default: console.error("Unsupported texture", texture.format.toString(16));
        }
    }

    texture.dstFormat = calcTextureDestFormat(texture);

    if (srcOffs !== null)
        convertTexturePixels();

    if (!texture.pixels) {
        if (texture.dstFormat === "i8")
            texture.pixels = new Uint8Array(texture.width * texture.height);
        else if (texture.dstFormat === "i8_a8")
            texture.pixels = new Uint8Array(texture.width * texture.height * 2);
        else if (texture.dstFormat === "rgba8")
            texture.pixels = new Uint8Array(texture.width * texture.height * 4);
    }

    function translateWrap(cm: number) {
        switch (cm) {
            case 1: return gl.MIRRORED_REPEAT;
            case 2: return gl.CLAMP_TO_EDGE;
            case 3: return gl.CLAMP_TO_EDGE;
            default: return gl.REPEAT;
        }
    }

    const texId = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texId);
    // Filters are set to NEAREST here because filtering is performed in the fragment shader.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, translateWrap(texture.cms));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, translateWrap(texture.cmt));

    let glFormat;
    if (texture.dstFormat === "i8")
        glFormat = gl.LUMINANCE;
    else if (texture.dstFormat === "i8_a8")
        glFormat = gl.LUMINANCE_ALPHA;
    else if (texture.dstFormat === "rgba8")
        glFormat = gl.RGBA;

    gl.texImage2D(gl.TEXTURE_2D, 0, glFormat, texture.width, texture.height, 0, glFormat, gl.UNSIGNED_BYTE, texture.pixels);
    texture.glTextureId = texId;
}

function calcTextureDestFormat(texture: TextureTile): TextureDestFormat {
    switch (texture.format & 0xE0) {
    case 0x00: return "rgba8"; // RGBA
    case 0x40: return "rgba8"; // CI -- XXX -- do we need to check the palette type?
    case 0x60: return "i8_a8"; // IA
    case 0x80: return "i8_a8"; // I
    default: throw new Error("Invalid texture type");
    }
}

function calcTextureSize(texture: TextureTile) {
    let maxTexel, lineShift;
    switch (texture.format) {
    // 4-bit
    case 0x00: maxTexel = 4096; lineShift = 4; break; // RGBA
    case 0x40: maxTexel = 4096; lineShift = 4; break; // CI
    case 0x60: maxTexel = 8196; lineShift = 4; break; // IA
    case 0x80: maxTexel = 8196; lineShift = 4; break; // I
    // 8-bit
    case 0x08: maxTexel = 2048; lineShift = 3; break; // RGBA
    case 0x48: maxTexel = 2048; lineShift = 3; break; // CI
    case 0x68: maxTexel = 4096; lineShift = 3; break; // IA
    case 0x88: maxTexel = 4096; lineShift = 3; break; // I
    // 16-bit
    case 0x10: maxTexel = 2048; lineShift = 2; break; // RGBA
    case 0x50: maxTexel = 2048; lineShift = 0; break; // CI
    case 0x70: maxTexel = 2048; lineShift = 2; break; // IA
    case 0x90: maxTexel = 2048; lineShift = 0; break; // I
    // 32-bit
    case 0x18: maxTexel = 1024; lineShift = 2; break; // RGBA
    default:
        throw "whoops";
    }

    const lineW = texture.lineSize << lineShift;
    const tileW = texture.lrs - texture.uls + 1;
    const tileH = texture.lrt - texture.ult + 1;

    const maskW = 1 << texture.maskS;
    const maskH = 1 << texture.maskT;

    let lineH;
    if (lineW > 0)
        lineH = Math.min(maxTexel / lineW, tileH);
    else
        lineH = 0;

    let width;
    if (texture.maskS > 0 && (maskW * maskH) <= maxTexel)
        width = maskW;
    else if ((tileW * tileH) <= maxTexel)
        width = tileW;
    else
        width = lineW;

    let height;
    if (texture.maskT > 0 && (maskW * maskH) <= maxTexel)
        height = maskH;
    else if ((tileW * tileH) <= maxTexel)
        height = tileH;
    else
        height = lineH;

    texture.width = width;
    texture.height = height;
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

    state.textureTiles = [];

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
