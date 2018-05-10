
// tslint:disable:no-console

import { mat4, vec3, vec4 } from 'gl-matrix';

import { RenderState, RenderFlags, Program, ColorTarget, DepthTarget } from './render';
import * as UI from './ui';

import Progressable from 'Progressable';

export interface CameraController {
    serialize(): string;
    deserialize(state: string): void;
    setInitialCamera(camera: mat4): void;
    update(camera: mat4, inputManager: InputManager, dt: number): boolean;
}

// XXX: Is there any way to do this properly and reference the interface?
export type CameraControllerClass = typeof FPSCameraController | typeof OrbitCameraController;

export interface Texture {
    name: string;
    surfaces: HTMLCanvasElement[];
}

export interface Scene {
    textures: Texture[];
    render(state: RenderState): void;
    destroy(gl: WebGL2RenderingContext): void;
}

class InputManager {
    public toplevel: HTMLElement;
    public keysDown: Map<string, boolean>;
    public dx: number;
    public dy: number;
    public dz: number;
    public button: number;
    private lastX: number;
    private lastY: number;
    public grabbing: boolean = false;

    constructor(toplevel: HTMLElement) {
        this.toplevel = toplevel;

        this.keysDown = new Map<string, boolean>();
        window.addEventListener('keydown', this._onKeyDown);
        window.addEventListener('keyup', this._onKeyUp);
        this.toplevel.addEventListener('wheel', this._onWheel, { passive: false });
        this.toplevel.addEventListener('mousedown', this._onMouseDown);

        this.resetMouse();
    }

    public isKeyDown(key: string) {
        return this.keysDown.get(key);
    }
    public isDragging(): boolean {
        return this.grabbing;
    }
    public resetMouse() {
        this.dx = 0;
        this.dy = 0;
        this.dz = 0;
    }

    private _onKeyDown = (e: KeyboardEvent) => {
        this.keysDown.set(e.code, true);
    };
    private _onKeyUp = (e: KeyboardEvent) => {
        this.keysDown.delete(e.code);
    };

    private _onWheel = (e: WheelEvent) => {
        e.preventDefault();
        this.dz += Math.sign(e.deltaY) * -4;
    };

    private _setGrabbing(v: boolean) {
        if (this.grabbing === v)
            return;

        this.grabbing = v;
        this.toplevel.style.cursor = v ? '-webkit-grabbing' : '-webkit-grab';
        this.toplevel.style.cursor = v ? 'grabbing' : 'grab';
        this.toplevel.style.setProperty('pointer-events', v ? 'auto' : '', 'important');

        if (v) {
            document.addEventListener('mousemove', this._onMouseMove);
            document.addEventListener('mouseup', this._onMouseUp);
        } else {
            document.removeEventListener('mousemove', this._onMouseMove);
            document.removeEventListener('mouseup', this._onMouseUp);
        }
    }

    private _onMouseMove = (e: MouseEvent) => {
        if (!this.grabbing)
            return;
        const dx = e.pageX - this.lastX;
        const dy = e.pageY - this.lastY;
        this.lastX = e.pageX;
        this.lastY = e.pageY;
        this.dx += dx;
        this.dy += dy;
    };
    private _onMouseUp = (e: MouseEvent) => {
        this._setGrabbing(false);
        this.button = 0;
    }
    private _onMouseDown = (e: MouseEvent) => {
        this.button = e.button;
        this.lastX = e.pageX;
        this.lastY = e.pageY;
        this._setGrabbing(true);
        // Needed to make the cursor update in Chrome. See:
        // https://bugs.chromium.org/p/chromium/issues/detail?id=676644
        this.toplevel.focus();
        e.preventDefault();
    }
}

const vec4Scratch = vec4.create();
export class FPSCameraController implements CameraController {
    private tmp: mat4;
    private camera: mat4;
    private speed: number;

    constructor() {
        this.tmp = mat4.create();
        this.camera = mat4.create();
        this.speed = 10;
    }

    public serialize(): string {
        const tx = this.camera[12], ty = this.camera[13], tz = this.camera[14];
        const rx = this.camera[0], ry = this.camera[4], rz = this.camera[8];
        const fx = this.camera[2], fy = this.camera[6], fz = this.camera[10];
        return `${tx.toFixed(2)},${ty.toFixed(2)},${tz.toFixed(2)},${fx.toFixed(2)},${fy.toFixed(2)},${fz.toFixed(2)},${rx.toFixed(2)},${ry.toFixed(2)},${rz.toFixed(2)}`;
    }

    public deserialize(state: string) {
        const [tx, ty, tz, fx, fy, fz, rx, ry, rz] = state.split(',');
        this.camera[12] = +tx;
        this.camera[13] = +ty;
        this.camera[14] = +tz;
        this.camera[2] = +fx;
        this.camera[6] = +fy;
        this.camera[10] = +fz;
        this.camera[0] = +rx;
        this.camera[4] = +ry;
        this.camera[8] = +rz;
        const u = vec3.create();
        vec3.cross(u, [this.camera[2], this.camera[6], this.camera[10]], [this.camera[0], this.camera[4], this.camera[8]]);
        vec3.normalize(u, u);
        this.camera[1] = u[0];
        this.camera[5] = u[1];
        this.camera[9] = u[2];
    }

    public setInitialCamera(camera: mat4) {
        mat4.invert(this.camera, camera);
    }

    public update(outCamera: mat4, inputManager: InputManager, dt: number): boolean {
        const tmp = this.tmp;
        const camera = this.camera;
        let updated = false;

        this.speed += inputManager.dz;
        this.speed = Math.max(this.speed, 1);

        let mult = this.speed;
        if (inputManager.isKeyDown('ShiftLeft') || inputManager.isKeyDown('ShiftRight'))
            mult *= 5;
        mult *= (dt / 16.0);

        let amt;
        amt = 0;
        if (inputManager.isKeyDown('KeyW')) {
            amt = -mult;
        } else if (inputManager.isKeyDown('KeyS')) {
            amt = mult;
        }
        updated = updated || (amt !== 0);
        tmp[14] = amt;

        amt = 0;
        if (inputManager.isKeyDown('KeyA')) {
            amt = -mult;
        } else if (inputManager.isKeyDown('KeyD')) {
            amt = mult;
        }
        updated = updated || (amt !== 0);
        tmp[12] = amt;

        amt = 0;
        if (inputManager.isKeyDown('KeyQ')) {
            amt = -mult;
        } else if (inputManager.isKeyDown('KeyE')) {
            amt = mult;
        }
        updated = updated || (amt !== 0);
        tmp[13] = amt;

        if (inputManager.isKeyDown('KeyB')) {
            mat4.identity(camera);
            updated = true;
        }
        if (inputManager.isKeyDown('KeyC')) {
            console.log(camera);
        }

        updated = updated || (inputManager.dx !== 0);
        updated = updated || (inputManager.dy !== 0);

        if (updated) {
            const cu = vec3.fromValues(camera[1], camera[5], camera[9]);
            vec3.normalize(cu, cu);
            mat4.rotate(camera, camera, -inputManager.dx / 500, cu);
            mat4.rotate(camera, camera, -inputManager.dy / 500, [1, 0, 0]);

            mat4.multiply(camera, camera, tmp);
        }

        // XXX: Is there any way to do this without the expensive inverse?
        mat4.invert(outCamera, camera);

        return updated;
    }
}

function clamp(v: number, min: number, max: number): number {
    return Math.max(min, Math.min(v, max));
}

function clampRange(v: number, lim: number): number {
    return clamp(v, -lim, lim);
}

export class OrbitCameraController implements CameraController {
    public x: number = 0.15;
    public y: number = 0.35;
    public z: number = -150;
    public xVel: number = 0;
    public yVel: number = 0;
    public zVel: number = 0;

    public tx: number = 0;
    public ty: number = 0;
    public txVel: number = 0;
    public tyVel: number = 0;

    constructor() {
    }

    public serialize(): string {
        return '';
    }

    public deserialize(state: string): void {
    }

    public setInitialCamera(camera: mat4) {
        // TODO(jstpierre)
    }

    public update(camera: mat4, inputManager: InputManager, dt: number): boolean {
        // Get new velocities from inputs.
        if (inputManager.button === 1) {
            this.txVel += inputManager.dx * (-10 - Math.min(this.z, 0.01)) /  5000;
            this.tyVel += inputManager.dy * (-10 - Math.min(this.z, 0.01)) / -5000;
        } else {
            this.xVel += inputManager.dx / 200;
            this.yVel += inputManager.dy / 200;
        }
        this.zVel += inputManager.dz;
        if (inputManager.isKeyDown('A')) {
            this.xVel += 0.05;
        }
        if (inputManager.isKeyDown('D')) {
            this.xVel -= 0.05;
        }
        if (inputManager.isKeyDown('W')) {
            this.yVel += 0.05;
        }
        if (inputManager.isKeyDown('S')) {
            this.yVel -= 0.05;
        }
        this.xVel = clampRange(this.xVel, 2);
        this.yVel = clampRange(this.yVel, 2);

        const updated = this.xVel !== 0 || this.yVel !== 0 || this.zVel !== 0;
        if (updated) {
            // Apply velocities.
            const drag = inputManager.isDragging() ? 0.92 : 0.96;

            this.x += this.xVel / 10;
            this.xVel *= drag;

            this.y += this.yVel / 10;
            this.yVel *= drag;

            this.tx += this.txVel;
            this.txVel *= drag;

            this.ty += this.tyVel;
            this.tyVel *= drag;

            if (this.y < 0.04) {
                this.y = 0.04;
                this.yVel = 0;
            }
            if (this.y > 1.50) {
                this.y = 1.50;
                this.yVel = 0;
            }

            this.z += this.zVel;
            this.zVel *= 0.8;
            if (this.z > -10) {
                this.z = -10;
                this.zVel = 0;
            }
        }

        // Calculate new camera from new x/y/z.
        const sinX = Math.sin(this.x);
        const cosX = Math.cos(this.x);
        const sinY = Math.sin(this.y);
        const cosY = Math.cos(this.y);
        mat4.copy(camera, mat4.fromValues(
            cosX, sinY * sinX, -cosY * sinX, 0,
            0, cosY, sinY, 0,
            sinX, -sinY * cosX, cosY * cosX, 0,
            this.tx, this.ty, this.z, 1,
        ));

        return updated;
    }
}

export class Viewer {
    public camera: mat4;
    public inputManager: InputManager;
    public cameraController: CameraController;

    public renderState: RenderState;
    private onscreenColorTarget: ColorTarget = new ColorTarget();
    private onscreenDepthTarget: DepthTarget = new DepthTarget();
    public scene: MainScene;
    
    public oncamerachanged: () => void = (() => {});

    constructor(public canvas: HTMLCanvasElement) {
        const gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
        this.renderState = new RenderState(gl);

        this.inputManager = new InputManager(this.canvas);

        this.camera = mat4.create();
        this.cameraController = null;
    }

    public reset() {
        const gl = this.renderState.gl;
        gl.activeTexture(gl.TEXTURE0);
        gl.clearColor(0.88, 0.88, 0.88, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        this.renderState.setClipPlanes(0.2, 50000);
    }

    public render() {
        const gl = this.renderState.gl;

        if (!this.scene)
            return;

        this.onscreenColorTarget.setParameters(gl, this.canvas.width, this.canvas.height);
        this.onscreenDepthTarget.setParameters(gl, this.canvas.width, this.canvas.height);
        this.renderState.setOnscreenRenderTarget(this.onscreenColorTarget, this.onscreenDepthTarget);
        this.renderState.reset();

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // Main scene. This renders to the onscreen target.
        this.scene.render(this.renderState);

        // Blit to the screen.
        this.renderState.blitOnscreenToGL();

        const frameEndTime = window.performance.now();
        const diff = frameEndTime - this.renderState.frameStartTime;
        // console.log(`Time: ${diff} Draw calls: ${state.drawCallCount}`);
    }

    public setCameraController(cameraController: CameraController) {
        this.cameraController = cameraController;
    }

    public setScene(scene: MainScene) {
        const gl = this.renderState.gl;

        this.reset();

        if (this.scene) {
            this.scene.destroy(gl);
        }

        if (scene) {
            this.scene = scene;
            this.oncamerachanged();
        } else {
            this.scene = null;
        }
    }

    public start() {
        const camera = this.camera;
        const canvas = this.canvas;

        let t = 0;
        const update = (nt: number) => {
            const dt = nt - t;
            t = nt;

            if (this.cameraController) {
                const updated = this.cameraController.update(camera, this.inputManager, dt);
                if (updated)
                    this.oncamerachanged();
            }

            this.inputManager.resetMouse();

            this.renderState.setView(camera);
            this.renderState.time += dt;
            this.render();

            window.requestAnimationFrame(update);
        };
        update(0);
    }
}

export interface MainScene extends Scene {
    resetCamera?(cameraController: CameraController): void;
    createPanels?(): UI.Panel[];
}

export interface SceneDesc {
    id: string;
    name: string;
    createScene(gl: WebGL2RenderingContext): Progressable<MainScene>;
    defaultCameraController?: CameraControllerClass;
}

export interface SceneGroup {
    id: string;
    name: string;
    sceneDescs: SceneDesc[];
}
