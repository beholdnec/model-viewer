
import * as F3DEX2 from './f3dex2';
import * as ZELVIEW0 from './zelview0';

import Progressable from 'Progressable';
import { CullMode, RenderFlags, RenderState, BlendMode } from '../render';
import Program from '../Program';
import { fetch } from '../util';

import * as Viewer from '../viewer';
import ArrayBufferSlice from 'ArrayBufferSlice';

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

class CombinerUniforms {
    public A: WebGLUniformLocation;
    public B: WebGLUniformLocation;
    public C: WebGLUniformLocation;
    public D: WebGLUniformLocation;
}

export class F3DEX2Program extends Program {
    public txsLocation: WebGLUniformLocation;
    public useVertexColorsLocation: WebGLUniformLocation;
    public alphaTestLocation: WebGLUniformLocation;
    public colorCombiners: Array<CombinerUniforms>;
    public alphaCombiners: Array<CombinerUniforms>;
    static a_Position = 0;
    static a_UV = 1;
    static a_Color = 2;

    public vert = `
uniform mat4 u_modelView;
uniform mat4 u_projection;
layout(location = ${F3DEX2Program.a_Position}) attribute vec3 a_Position;
layout(location = ${F3DEX2Program.a_UV}) attribute vec2 a_UV;
layout(location = ${F3DEX2Program.a_Color}) attribute vec4 a_Color;
out vec4 v_color;
out vec2 v_uv;
uniform vec2 u_txs;

void main() {
    gl_Position = u_projection * u_modelView * vec4(a_Position, 1.0);
    v_uv = a_UV * u_txs;
    v_color = a_Color;
}
`;

    public frag = `
precision mediump float;
varying vec2 v_uv;
varying vec4 v_color;
uniform sampler2D u_texture;
uniform bool u_useVertexColors;
uniform int u_alphaTest;
struct Combiner {
    int A;
    int B;
    int C;
    int D;
};
uniform Combiner u_colorCombiners[2];
uniform Combiner u_alphaCombiners[2];

vec4 n64Texture2D(sampler2D tex, vec2 texCoord) {
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 offset = fract(texCoord * texSize - 0.5);
    offset -= step(1.0, offset.x + offset.y);
    vec4 c0 = texture2D(tex, texCoord - offset / texSize, 0.0);
    vec4 c1 = texture2D(tex, texCoord - vec2(offset.x - sign(offset.x), offset.y) / texSize, 0.0);
    vec4 c2 = texture2D(tex, texCoord - vec2(offset.x, offset.y - sign(offset.y)) / texSize, 0.0);
    return c0 + abs(offset.x) * (c1 - c0) + abs(offset.y) * (c2 - c0);		
}

vec3 getSubAColor(vec4 combined, int mode) {
    vec3 result = vec3(0.0);
    switch (mode) {
    case 0: // COMBINED
        result = combined.rgb;
        break;
    case 1: // TEXEL0
        result = n64Texture2D(u_texture, v_uv).rgb;
        break;
    case 4: // SHADE
        result = vec3(1.0); // TODO
        break;
    default:
        result = vec3(0.0);
        break;
    }
    return result;
}

vec3 getSubBColor(vec4 combined, int mode) {
    vec3 result = vec3(0.0);
    switch (mode) {
    case 1: // TEXEL0
        result = n64Texture2D(u_texture, v_uv).rgb;
        break;
    default:
        result = vec3(0.0);
        break;
    }
    return result;
}

vec3 getMulColor(vec4 combined, int mode) {
    vec3 result = vec3(0.0);
    switch (mode) {
    case 3: // PRIMITIVE
        result = v_color.rgb; // TODO: Implement u_useVertexColors option
        break;
    case 4: // SHADE
        result = vec3(1.0); // TODO
        break;
    case 12: // ENV_ALPHA
        result = vec3(1.0); // TODO
        break;
    default:
        result = vec3(0.0);
        break;
    }
    return result;
}

vec3 getAddColor(vec4 combined, int mode) {
    vec3 result = vec3(0.0);
    switch (mode) {
    case 0: // COMBINED
        result = combined.rgb;
        break;
    default:
        result = vec3(0.0);
        break;
    }
    return result;
}

float getMulAlpha(vec4 combined, int mode) {
    float result = 0.0;
    switch (mode) {
    default:
        result = 0.0;
        break;
    }
    return result;
}

float getAddSubAlpha(vec4 combined, int mode) {
    float result = 0.0;
    switch (mode) {
    case 0: // COMBINED
        result = combined.a;
        break;
    case 1: // TEXEL0
        result = n64Texture2D(u_texture, v_uv).a;
        break;
    case 6: // 1
        result = 1.0;
        break;
    default:
        result = 0.0;
        break;
    }
    return result;
}

vec4 combine(vec4 combined, int combiner) {
    vec3 cA = getSubAColor(combined, u_colorCombiners[combiner].A);
    vec3 cB = getSubBColor(combined, u_colorCombiners[combiner].B);
    vec3 cC = getMulColor(combined, u_colorCombiners[combiner].C);
    vec3 cD = getAddColor(combined, u_colorCombiners[combiner].D);
    vec3 c = (cA - cB) * cC + cD;
    float aA = getAddSubAlpha(combined, u_alphaCombiners[combiner].A);
    float aB = getAddSubAlpha(combined, u_alphaCombiners[combiner].B);
    float aC = getMulAlpha(combined, u_alphaCombiners[combiner].C);
    float aD = getAddSubAlpha(combined, u_alphaCombiners[combiner].D);
    float a = (aA - aB) * aC + aD;
    return vec4(c, a);
}

void main() {
    gl_FragColor = combine(vec4(0.0), 1);
    // TODO: Don't perform second combine if combiner is in 1-cycle mode.
    gl_FragColor = combine(gl_FragColor, 0);
    //if (u_useVertexColors)
    //    gl_FragColor *= v_color;
    if (u_alphaTest > 0 && gl_FragColor.a < 0.0125)
        discard;
}
`;

    public bind(gl: WebGL2RenderingContext, prog: WebGLProgram) {
        super.bind(gl, prog);

        this.txsLocation = gl.getUniformLocation(prog, "u_txs");
        this.useVertexColorsLocation = gl.getUniformLocation(prog, "u_useVertexColors");
        this.alphaTestLocation = gl.getUniformLocation(prog, "u_alphaTest");
        this.colorCombiners = new Array(2);
        this.alphaCombiners = new Array(2);
        for (let i = 0; i < 2; i++) {
            this.colorCombiners[i] = {
                A: gl.getUniformLocation(prog, `u_colorCombiners[${i}].A`),
                B: gl.getUniformLocation(prog, `u_colorCombiners[${i}].B`),
                C: gl.getUniformLocation(prog, `u_colorCombiners[${i}].C`),
                D: gl.getUniformLocation(prog, `u_colorCombiners[${i}].D`),
            };
            this.alphaCombiners[i] = {
                A: gl.getUniformLocation(prog, `u_alphaCombiners[${i}].A`),
                B: gl.getUniformLocation(prog, `u_alphaCombiners[${i}].B`),
                C: gl.getUniformLocation(prog, `u_alphaCombiners[${i}].C`),
                D: gl.getUniformLocation(prog, `u_alphaCombiners[${i}].D`),
            };
            
        }
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

class Scene implements Viewer.MainScene {
    public textures: Viewer.Texture[];
    public zelview0: ZELVIEW0.ZELVIEW0;
    public program_BG: BillboardBGProgram;
    public program_COLL: CollisionProgram;
    public program_DL: F3DEX2Program;
    public program_WATERS: WaterboxProgram;

    public render: RenderFunc;

    constructor(gl: WebGL2RenderingContext, zelview0: ZELVIEW0.ZELVIEW0) {
        this.zelview0 = zelview0;
        this.textures = [];
        this.program_BG = new BillboardBGProgram();
        this.program_COLL = new CollisionProgram();
        this.program_DL = new F3DEX2Program();
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

                state.useProgram(this.program_DL);
                state.bindModelView();
                mesh.opaque.forEach(renderDL);
                mesh.transparent.forEach(renderDL);
            };

            const renderRoom = (room: ZELVIEW0.Headers) => {
                renderMesh(room.mesh);
            };

            state.useProgram(this.program_DL);
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
