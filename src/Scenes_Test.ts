
import * as Viewer from "./viewer";
import { GfxDevice, GfxRenderPass } from "./gfx/platform/GfxPlatform";
import { IS_DEVELOPMENT } from "./BuildVersion";
import { SceneContext } from "./SceneBase";

import { createBasicRRESRendererFromBRRES } from "./rres/scenes";
import * as H3D from "./Common/CTR_H3D/H3D";
import * as PVRT from "./Common/DC/PVRT";
import { CtrTextureHolder } from "./oot3d/render";
import { decompress, ContentReader } from "./Fez/XNB";
import { FezContentTypeReaderManager } from "./Fez/XNB_Fez";

const id = 'test';
const name = "Test Scenes";

class BasicRRESSceneDesc implements Viewer.SceneDesc {
    constructor(public dataPath: string, public id: string = dataPath, public name: string = dataPath) {}

    public createScene(device: GfxDevice, context: SceneContext): Promise<Viewer.SceneGfx> {
        const dataFetcher = context.dataFetcher;
        return dataFetcher.fetchData(this.dataPath).then((data) => {
            return createBasicRRESRendererFromBRRES(device, [data]);
        });
    }
}

class H3DScene implements Viewer.SceneGfx {
    public textureHolder = new CtrTextureHolder();

    public render(device: GfxDevice, viewerInput: Viewer.ViewerRenderInput): GfxRenderPass {
        return null as unknown as GfxRenderPass;
    }

    public destroy(device: GfxDevice): void {
    }
}

class H3DSceneDesc implements Viewer.SceneDesc {
    constructor(public dataPath: string, public id: string = dataPath, public name: string = dataPath) {}

    public createScene(device: GfxDevice, context: SceneContext): Promise<Viewer.SceneGfx> {
        const dataFetcher = context.dataFetcher;
        return dataFetcher.fetchData(this.dataPath).then((data) => {
            const h3d = H3D.parse(data);
            const renderer = new H3DScene();
            renderer.textureHolder.addTextures(device, h3d.textures);
            return renderer;
        });
    }
}

export class JetSetRadioScene implements Viewer.SceneGfx {
    public textureHolder = new PVRT.PVRTextureHolder();
    
    public render(device: GfxDevice, viewerInput: Viewer.ViewerRenderInput): GfxRenderPass {
        return null as unknown as GfxRenderPass;
    }

    public destroy(device: GfxDevice): void {
    }
}

class XNBTest implements Viewer.SceneDesc {
    constructor(public dataPath: string, public id: string = dataPath, public name: string = dataPath) {}

    public createScene(device: GfxDevice, context: SceneContext): Promise<Viewer.SceneGfx> {
        const dataFetcher = context.dataFetcher;
        return dataFetcher.fetchData(this.dataPath).then((data) => {
            const decompressed = decompress(data);
            const typeReaderManager = new FezContentTypeReaderManager();
            const stream = new ContentReader(typeReaderManager, decompressed);
            const obj = stream.ReadAsset();
            console.log(obj);
            return new JetSetRadioScene();
        });
    }
}

const sceneDescs = [
    new BasicRRESSceneDesc('test/dthro_cmn1.brres'),
    new H3DSceneDesc('test/cave_Common.bch'),
    //new JetSetRadioSceneDesc('jsr/DPTEX/FLAGTEX001.PVR'),
    new XNBTest('test/1_bit_doorao.xnb'),
];

export const sceneGroup: Viewer.SceneGroup = {
    id, name, sceneDescs, hidden: !IS_DEVELOPMENT,
};
