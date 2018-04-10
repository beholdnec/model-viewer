
import * as F3DEX2 from './f3dex2';
import * as ZELVIEW0 from './zelview0';

import Progressable from 'Progressable';
import { CullMode, RenderFlags, RenderState, BlendMode } from '../render';
import Program from '../Program';
import { fetch } from '../util';

import * as Viewer from '../viewer';
import ArrayBufferSlice from 'ArrayBufferSlice';
import { PrimitiveType } from '../gx/gx_enum';

export type RenderFunc = (renderState: RenderState) => void;

export class BillboardBGProgram extends Program {
    public positionLocation: number;
    public uvLocation: number;

    public vert = `
attribute vec3 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;

void main() {
    gl_Position = vec4(a_position, 1.0);
    v_uv = a_uv;
}
`;
    public frag = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_texture;

void main() {
    gl_FragColor = texture2D(u_texture, v_uv);
}
`;

    public bind(gl: WebGL2RenderingContext, prog: WebGLProgram) {
        super.bind(gl, prog);

        this.positionLocation = gl.getAttribLocation(prog, "a_position");
        this.uvLocation = gl.getAttribLocation(prog, "a_uv");
    }
}

interface Combiner {
    subA: number;
    subB: number;
    mul: number;
    add: number;
}

interface F3DEX2ProgramParameters {
    use2Cycle: boolean;
    colorCombiners: Array<Combiner>;
    alphaCombiners: Array<Combiner>;
}

const F3DEX2_FRAG_BASE = `
precision mediump float;
varying vec2 v_uv;
varying vec4 v_color;
uniform sampler2D u_texture0;
uniform sampler2D u_texture1;
uniform bool u_useVertexColors;
uniform int u_alphaTest;

vec4 n64Texture2D(sampler2D tex, vec2 texCoord) {
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 offset = fract(texCoord * texSize - 0.5);
    offset -= step(1.0, offset.x + offset.y);
    vec4 c0 = texture2D(tex, texCoord - offset / texSize, 0.0);
    vec4 c1 = texture2D(tex, texCoord - vec2(offset.x - sign(offset.x), offset.y) / texSize, 0.0);
    vec4 c2 = texture2D(tex, texCoord - vec2(offset.x, offset.y - sign(offset.y)) / texSize, 0.0);
    return c0 + abs(offset.x) * (c1 - c0) + abs(offset.y) * (c2 - c0);		
}

void main() {
    vec4 t0 = n64Texture2D(u_texture0, v_uv);
    vec4 t1 = n64Texture2D(u_texture1, v_uv);

    vec4 combined = vec4(0.0);
#if USE_2CYCLE
    combined.rgb = (CC0_SUBA - CC0_SUBB) * CC0_MUL + CC0_ADD;
    combined.a = (AC0_SUBA - AC0_SUBB) * AC0_MUL + AC0_ADD;
#endif
    combined.rgb = (CC1_SUBA - CC1_SUBB) * CC1_MUL + CC1_ADD;
    combined.a = (AC1_SUBA - AC1_SUBB) * AC1_MUL + AC1_ADD;

    gl_FragColor = combined;

    if (u_alphaTest > 0 && gl_FragColor.a < 0.0125)
        discard;
}
`;

const CC_SUBA: {[mode: number]: string} = {
    0: `combined.rgb`, // COMBINED
    1: `t0.rgb`, // TEXEL0
    2: `t1.rgb`, // TEXEL1
    3: `v_color.rgb`, // PRIMITIVE
    4: `vec3(1.0)`, // SHADE (TODO)
    5: `vec3(1.0)`, // ENVIRONMENT (TODO)
    6: `vec3(1.0)`, // 1
    7: `vec3(0.5)`, // NOISE (TODO)
};

const CC_SUBB: {[mode: number]: string} = {
    0: `combined.rgb`, // COMBINED
    1: `t0.rgb`, // TEXEL0
    2: `t1.rgb`, // TEXEL1
    3: `v_color.rgb`, // PRIMITIVE
    4: `vec3(1.0)`, // SHADE (TODO)
    5: `vec3(1.0)`, // ENVIRONMENT (TODO)
    6: `vec3(0.0)`, // CENTER (i.e. key-center) (TODO)
    7: `vec3(0.5)`, // K4 (TODO)
};

const CC_MUL: {[mode: number]: string} = {
    0: `combined.rgb`, // COMBINED
    1: `t0.rgb`, // TEXEL0
    2: `t1.rgb`, // TEXEL1
    3: `v_color.rgb`, // PRIMITIVE
    4: `vec3(1.0)`, // SHADE (TODO)
    5: `vec3(1.0)`, // ENVIRONMENT (TODO)
    6: `vec3(1.0)`, // SCALE (i.e. key-scale) (TODO)
    7: `combined.aaa`, // COMBINED_ALPHA
    8: `t0.aaa`, // TEXEL0_ALPHA
    9: `t1.aaa`, // TEXEL1_ALPHA
    10: `v_color.aaa`, // PRIMITIVE_ALPHA
    11: `vec3(1.0)`, // SHADE_ALPHA (TODO)
    12: `vec3(1.0)`, // ENV_ALPHA (TODO)
    13: `vec3(1.0)`, // LOD_FRACTION (TODO)
    14: `vec3(1.0)`, // PRIM_LOD_FRAC (TODO)
    15: `vec3(0.5)`, // K5 (TODO)
};

const CC_ADD: {[mode: number]: string} = {
    0: `combined.rgb`, // COMBINED
    1: `t0.rgb`, // TEXEL0
    2: `t1.rgb`, // TEXEL1
    3: `v_color.rgb`, // PRIMITIVE
    4: `vec3(1.0)`, // SHADE (TODO)
    5: `vec3(1.0)`, // ENVIRONMENT (TODO)
    6: `vec3(1.0)`, // 1
};

const AC_ADDSUB: {[mode: number]: string} = {
    0: `combined.a`, // COMBINED
    1: `t0.a`, // TEXEL0
    2: `t1.a`, // TEXEL1
    3: `v_color.a`, // PRIMITIVE
    4: `1.0`, // SHADE (TODO)
    5: `1.0`, // ENVIRONMENT (TODO)
    6: `1.0`, // 1
};

const AC_MUL: {[mode: number]: string} = {
    0: `1.0`, // LOD_FRACTION (TODO)
    1: `t0.a`, // TEXEL0
    2: `t1.a`, // TEXEL1
    3: `v_color.a`, // PRIMITIVE
    4: `1.0`, // SHADE (TODO)
    5: `1.0`, // ENVIRONMENT (TODO)
    6: `1.0`, // PRIM_LOD_FRAC (TODO)
};

function getOrDefault(obj: {[k: number]: any}, key: number, def: any) {
    const result = obj[key];
    return (result === undefined) ? def : result;
}

export class F3DEX2Program extends Program {
    public texture0Location: WebGLUniformLocation;
    public texture1Location: WebGLUniformLocation;
    public txsLocation: Array<WebGLUniformLocation>;
    public useVertexColorsLocation: WebGLUniformLocation;
    public alphaTestLocation: WebGLUniformLocation;
    static a_Position = 0;
    static a_UV = 1;
    static a_Color = 2;

    constructor(params: F3DEX2ProgramParameters) {
        super();

        this.frag = `#define USE_2CYCLE ${params.use2Cycle ? 1 : 0}\n`;

        for (let i = 0; i < 2; i++) {
            this.frag += `
#define CC${i}_SUBA ${getOrDefault(CC_SUBA, params.colorCombiners[i].subA, 'vec3(0.0)')}
#define CC${i}_SUBB ${getOrDefault(CC_SUBB, params.colorCombiners[i].subB, 'vec3(0.0)')}
#define CC${i}_MUL ${getOrDefault(CC_MUL, params.colorCombiners[i].mul, 'vec3(0.0)')}
#define CC${i}_ADD ${getOrDefault(CC_ADD, params.colorCombiners[i].add, 'vec3(0.0)')}
#define AC${i}_SUBA ${getOrDefault(AC_ADDSUB, params.alphaCombiners[i].subA, '0.0')}
#define AC${i}_SUBB ${getOrDefault(AC_ADDSUB, params.alphaCombiners[i].subB, '0.0')}
#define AC${i}_MUL ${getOrDefault(AC_MUL, params.alphaCombiners[i].mul, '0.0')}
#define AC${i}_ADD ${getOrDefault(AC_ADDSUB, params.alphaCombiners[i].add, '0.0')}
`;
        }

        this.frag += F3DEX2_FRAG_BASE;
    }

    public vert = `
uniform mat4 u_modelView;
uniform mat4 u_projection;
layout(location = ${F3DEX2Program.a_Position}) attribute vec3 a_Position;
layout(location = ${F3DEX2Program.a_UV}) attribute vec2 a_UV;
layout(location = ${F3DEX2Program.a_Color}) attribute vec4 a_Color;
out vec4 v_color;
out vec2 v_uv;
uniform vec2 u_txs[2];

void main() {
    gl_Position = u_projection * u_modelView * vec4(a_Position, 1.0);
    v_uv = a_UV * u_txs[0]; // ??? Is there a second set of texcoords?
    v_color = a_Color;
}
`;

    public frag = `
#error Shader was not properly constructed.
`

    public bind(gl: WebGL2RenderingContext, prog: WebGLProgram) {
        super.bind(gl, prog);

        this.texture0Location = gl.getUniformLocation(prog, "u_texture0");
        this.texture1Location = gl.getUniformLocation(prog, "u_texture1");
        this.txsLocation = new Array(2);
        this.txsLocation[0] = gl.getUniformLocation(prog, "u_txs[0]");
        this.txsLocation[1] = gl.getUniformLocation(prog, "u_txs[1]");
        this.useVertexColorsLocation = gl.getUniformLocation(prog, "u_useVertexColors");
        this.alphaTestLocation = gl.getUniformLocation(prog, "u_alphaTest");
    }
}

class CollisionProgram extends Program {
    public positionLocation: number;

    public vert = `
uniform mat4 u_modelView;
uniform mat4 u_projection;
attribute vec3 a_position;

void main() {
    gl_Position = u_projection * u_modelView * vec4(a_position, 1.0);
}
`;
    public frag = `
#extension GL_EXT_frag_depth : enable

void main() {
    gl_FragColor = vec4(1.0, 1.0, 1.0, 0.2);
    gl_FragDepthEXT = gl_FragCoord.z - 1e-6;
}
`;

    public bind(gl: WebGL2RenderingContext, prog: WebGLProgram) {
        super.bind(gl, prog);

        this.positionLocation = gl.getAttribLocation(prog, "a_position");
    }
}

class WaterboxProgram extends Program {
    public positionLocation: number;

    public vert = `
uniform mat4 u_modelView;
uniform mat4 u_projection;
attribute vec3 a_position;

void main() {
    gl_Position = u_projection * u_modelView * vec4(a_position, 1.0);
}
`;
    public frag = `
void main() {
    gl_FragColor = vec4(0.2, 0.6, 1.0, 0.2);
}
`;

    public bind(gl: WebGL2RenderingContext, prog: WebGLProgram) {
        super.bind(gl, prog);

        this.positionLocation = gl.getAttribLocation(prog, "a_position");
    }
}

function hashF3DEX2Params(params: F3DEX2ProgramParameters): string {
    return JSON.stringify(params); // TODO: use a more efficient hash mechanism
}

// F3DEX2-specific data stored in RenderState.
export interface F3DEX2UserState {
    progParams: F3DEX2ProgramParameters;
}

class Scene implements Viewer.MainScene {
    public textures: Viewer.Texture[];
    public zelview0: ZELVIEW0.ZELVIEW0;
    public program_BG: BillboardBGProgram;
    public program_COLL: CollisionProgram;
    public programMap_DL: {[hash: string]: F3DEX2Program};
    public program_WATERS: WaterboxProgram;

    public render: RenderFunc;

    constructor(gl: WebGL2RenderingContext, zelview0: ZELVIEW0.ZELVIEW0) {
        this.zelview0 = zelview0;
        this.textures = [];
        this.program_BG = new BillboardBGProgram();
        this.program_COLL = new CollisionProgram();
        this.programMap_DL = {};
        this.program_WATERS = new WaterboxProgram();

        const mainScene = zelview0.loadMainScene(gl);
        mainScene.rooms.forEach((room) => {
            this.textures = this.textures.concat(room.mesh.textures);
        });

        const renderScene = this.translateScene(gl, mainScene);
        const renderCollision = this.translateCollision(gl, mainScene);
        const renderWaterBoxes = this.translateWaterBoxes(gl, mainScene);
        this.render = (state: RenderState) => {
            renderScene(state);
            //renderCollision(state);
            renderWaterBoxes(state);
        };
    }

    private getDLProgram(params: F3DEX2ProgramParameters): F3DEX2Program {
        const hash = hashF3DEX2Params(params);
        if (!(hash in this.programMap_DL)) {
            this.programMap_DL[hash] = new F3DEX2Program(params);
        }
        return this.programMap_DL[hash];
    }

    private translateScene(gl: WebGL2RenderingContext, scene: ZELVIEW0.Headers): (state: RenderState) => void {
        return (state: RenderState) => {
            const gl = state.gl;

            const renderDL = (dl: F3DEX2.DL) => {
                dl.render(state);
            };

            const renderMesh = (mesh: ZELVIEW0.Mesh) => {
                if (mesh.bg) {
                    state.useProgram(this.program_BG);
                    state.bindModelView();
                    mesh.bg(state);
                }

                const userState = <F3DEX2UserState> state.userState;
                // TODO: Don't call getDLProgram if state didn't change; it could be expensive.
                state.useProgram(this.getDLProgram(userState.progParams));
                state.bindModelView();
                mesh.opaque.forEach(renderDL);
                mesh.transparent.forEach(renderDL);
            };

            const renderRoom = (room: ZELVIEW0.Headers) => {
                renderMesh(room.mesh);
            };

            // Initialize with default program parameters
            const userState: F3DEX2UserState = {
                progParams: {
                    use2Cycle: true,
                    colorCombiners: [
                        {subA: 0, subB: 0, mul: 0, add: 0},
                        {subA: 0, subB: 0, mul: 0, add: 0},
                    ],
                    alphaCombiners: [
                        {subA: 0, subB: 0, mul: 0, add: 0},
                        {subA: 0, subB: 0, mul: 0, add: 0},
                    ],
                }
            };
            state.userState = userState;
            
            scene.rooms.forEach((room) => renderRoom(room));
        };
    }

    private translateCollision(gl: WebGL2RenderingContext, scene: ZELVIEW0.Headers): (state: RenderState) => void {
        const coll = scene.collision;

        function stitchLines(ibd: Uint16Array): Uint16Array {
            const lines = new Uint16Array(ibd.length * 2);
            let o = 0;
            for (let i = 0; i < ibd.length; i += 3) {
                lines[o++] = ibd[i + 0];
                lines[o++] = ibd[i + 1];
                lines[o++] = ibd[i + 1];
                lines[o++] = ibd[i + 2];
                lines[o++] = ibd[i + 2];
                lines[o++] = ibd[i + 0];
            }
            return lines;
        }
        const collIdxBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, collIdxBuffer);
        const lineData = stitchLines(coll.polys);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lineData, gl.STATIC_DRAW);
        const nLinePrim = lineData.length;

        const collVertBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, collVertBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, coll.verts, gl.STATIC_DRAW);

        const renderFlags = new RenderFlags();
        renderFlags.depthTest = true;
        renderFlags.blendMode = BlendMode.ADD;

        return (state: RenderState) => {
            const prog = this.program_COLL;
            state.useProgram(prog);
            state.bindModelView();
            state.useFlags(renderFlags);
            gl.bindBuffer(gl.ARRAY_BUFFER, collVertBuffer);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, collIdxBuffer);
            gl.vertexAttribPointer(prog.positionLocation, 3, gl.SHORT, false, 0, 0);
            gl.enableVertexAttribArray(prog.positionLocation);
            gl.drawElements(gl.LINES, nLinePrim, gl.UNSIGNED_SHORT, 0);
            gl.disableVertexAttribArray(prog.positionLocation);
        };
    }

    private translateWaterBoxes(gl: WebGL2RenderingContext, scene: ZELVIEW0.Headers): (state: RenderState) => void {
        const coll = scene.collision;

        const wbVtx = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, wbVtx);
        gl.bufferData(gl.ARRAY_BUFFER, coll.waters, gl.STATIC_DRAW);
        const wbIdxData = new Uint16Array(coll.waters.length / 3);
        for (let i = 0; i < wbIdxData.length; i++)
            wbIdxData[i] = i;
        const wbIdx = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wbIdx);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, wbIdxData, gl.STATIC_DRAW);

        const renderFlags = new RenderFlags();
        renderFlags.blendMode = BlendMode.ADD;
        renderFlags.cullMode = CullMode.NONE;

        return (state: RenderState) => {
            const prog = this.program_WATERS;
            state.useProgram(prog);
            state.bindModelView();
            state.useFlags(renderFlags);
            gl.bindBuffer(gl.ARRAY_BUFFER, wbVtx);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wbIdx);
            gl.vertexAttribPointer(prog.positionLocation, 3, gl.SHORT, false, 0, 0);
            gl.enableVertexAttribArray(prog.positionLocation);
            for (let i = 0; i < wbIdxData.length; i += 4)
                gl.drawElements(gl.TRIANGLE_STRIP, 4, gl.UNSIGNED_SHORT, i * 2);
            gl.disableVertexAttribArray(prog.positionLocation);
        };
    }

    public destroy(gl: WebGL2RenderingContext) {
        // TODO(jstpierre): Implement destroy for zelview.
    }
}

export class SceneDesc implements Viewer.SceneDesc {
    public id: string;
    public name: string;
    public path: string;

    constructor(name: string, path: string) {
        this.name = name;
        this.path = path;
        this.id = this.path;
    }

    public createScene(gl: WebGL2RenderingContext): Progressable<Scene> {
        return fetch(this.path).then((result: ArrayBufferSlice) => {
            const zelview0 = ZELVIEW0.readZELVIEW0(result);
            return new Scene(gl, zelview0);
        });
    }
}
