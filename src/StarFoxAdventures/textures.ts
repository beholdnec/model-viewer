import { hexzero } from '../util';
import ArrayBufferSlice from '../ArrayBufferSlice';
import * as GX_Texture from '../gx/gx_texture';
import { loadTextureFromMipChain, translateWrapModeGfx, translateTexFilterGfx } from '../gx/gx_render';
import { GfxDevice, GfxMipFilterMode, GfxTexture, GfxSampler, GfxFormat, makeTextureDescriptor2D, GfxWrapMode, GfxTexFilterMode } from '../gfx/platform/GfxPlatform';

import { loadRes } from './resource';

interface LoadedTexture {
    offset: number;
    texture: GX_Texture.TextureInputGX;
    wrapS: number;
    wrapT: number;
    minFilt: number;
    magFilt: number;
}

export interface DecodedTexture {
    gfxTexture: GfxTexture;
    gfxSampler: GfxSampler;
    width: number;
    height: number;
}

function loadTex(texData: ArrayBufferSlice, offset: number): LoadedTexture {
    const dv = texData.createDataView();
    const result = {
        offset,
        texture: {
            name: `Texture`,
            width: dv.getUint16(0x0A),
            height: dv.getUint16(0x0C),
            format: dv.getUint8(0x16),
            mipCount: dv.getUint16(0x1c) + 1,
            data: texData.slice(0x60),
        },
        wrapS: dv.getUint8(0x17),
        wrapT: dv.getUint8(0x18),
        minFilt: dv.getUint8(0x19),
        magFilt: dv.getUint8(0x1A),
    };
    return result;
}

function decodeTex(device: GfxDevice, loaded: LoadedTexture): DecodedTexture {
    const mipChain = GX_Texture.calcMipChain(loaded.texture, loaded.texture.mipCount);
    const gfxTexture = loadTextureFromMipChain(device, mipChain).gfxTexture;
    
    // GL texture is bound by loadTextureFromMipChain.
    const [minFilter, mipFilter] = translateTexFilterGfx(loaded.minFilt);
    const gfxSampler = device.createSampler({
        wrapS: translateWrapModeGfx(loaded.wrapS),
        wrapT: translateWrapModeGfx(loaded.wrapT),
        minFilter: minFilter, // TODO: implement mip filters
        magFilter: translateTexFilterGfx(loaded.magFilt)[0],
        mipFilter: mipFilter,
        minLOD: 0,
        maxLOD: 100,
    });

    return {
        gfxTexture,
        gfxSampler,
        width: loaded.texture.width,
        height: loaded.texture.height,
    };
}

function isValidTextureTabValue(tabValue: number) {
    return tabValue != 0xFFFFFFFF && (tabValue & 0x80000000) != 0;
}

function loadFirstValidTexture(device: GfxDevice, tab: ArrayBufferSlice, bin: ArrayBufferSlice): DecodedTexture | null {
    const tabDv = tab.createDataView();
    let firstValidId = 0;
    let found = false;
    for (let i = 0; i < tab.byteLength; i += 4) {
        const tabValue = tabDv.getUint32(i);
        if (tabValue == 0xFFFFFFFF) {
            console.log(`no valid id found`);
            break;
        }
        if (isValidTextureTabValue(tabValue)) {
            found = true;
            break;
        }
        ++firstValidId;
    }
    if (!found) {
        return null;
    }
    console.log(`loading first valid id ${firstValidId}`);
    return loadTextureFromTable(device, tab, bin, firstValidId);
}

function loadTextureFromTable(device: GfxDevice, tab: ArrayBufferSlice, bin: ArrayBufferSlice, id: number): (DecodedTexture | null) {
    const tabDv = tab.createDataView();
    const idOffs = id * 4;
    if (idOffs < 0 || idOffs + 4 >= tabDv.byteLength) {
        console.warn(`Texture id 0x${id.toString(16)} out of range; using first valid texture!`);
        return loadFirstValidTexture(device, tab, bin);
    }
    const tab0 = tabDv.getUint32(id * 4);
    if (isValidTextureTabValue(tab0)) {
        // Loadable texture (?)
        const binOffs = (tab0 & 0x00FFFFFF) * 2;
        const compData = bin.slice(binOffs);
        const uncompData = loadRes(compData);
        const loaded = loadTex(uncompData, (tab0 & 0x00FFFFFF) * 2);
        const decoded = decodeTex(device, loaded);
        return decoded;
    } else {
        // TODO: also seen is value 0x01000000
        console.warn(`Texture id 0x${id.toString(16)} (tab value 0x${hexzero(tab0, 8)}) not found in table. Using first valid texture.`);
        return loadFirstValidTexture(device, tab, bin);
    }
}

export abstract class TextureCollection {
    public abstract getTexture(device: GfxDevice, num: number): DecodedTexture | null;
}

function makeFalseTexture(device: GfxDevice, num: number): DecodedTexture {
    const gfxTexture = device.createTexture(makeTextureDescriptor2D(GfxFormat.U8_RGBA_NORM, 2, 2, 1));
    const gfxSampler = device.createSampler({
        wrapS: GfxWrapMode.REPEAT,
        wrapT: GfxWrapMode.REPEAT,
        minFilter: GfxTexFilterMode.BILINEAR,
        magFilter: GfxTexFilterMode.BILINEAR,
        mipFilter: GfxMipFilterMode.NO_MIP,
        minLOD: 0,
        maxLOD: 100,
    });


    // Thanks, StackOverflow.
    let seed = num;
    console.log(`making false texture with seed ${seed}`);
    function random() {
        let x = Math.sin(seed++) * 10000;
        return x - Math.floor(x);
    }

    const baseColor = [127 + random() * 127, 127 + random() * 127, 127 + random() * 127];
    const darkBase = [baseColor[0] * 0.7, baseColor[1] * 0.7, baseColor[2] * 0.7];
    const light = [baseColor[0], baseColor[1], baseColor[2], 0xff];
    const dark = [darkBase[0], darkBase[1], darkBase[2], 0xff];

    const pixels = new Uint8Array(4 * 4);
    pixels.set(dark, 0);
    pixels.set(light, 4);
    pixels.set(light, 8);
    pixels.set(dark, 12);

    const hostAccessPass = device.createHostAccessPass();
    hostAccessPass.uploadTextureData(gfxTexture, 0, [pixels]);
    device.submitPass(hostAccessPass);

    return {
        gfxTexture,
        gfxSampler,
        width: 2,
        height: 2,
    }
}

export class FalseTextureCollection implements TextureCollection {
    textures: DecodedTexture[] = [];

    constructor(device: GfxDevice) {
    }

    public getTexture(device: GfxDevice, num: number): DecodedTexture | null {
        if (this.textures[num] === undefined) {
            this.textures[num] = makeFalseTexture(device, num);
        }
        return this.textures[num];
    }
}

export class SFATextureCollection implements TextureCollection {
    decodedTextures: (DecodedTexture | null)[] = [];

    constructor(public tex1Tab: ArrayBufferSlice, public tex1Bin: ArrayBufferSlice) {
    }

    public getTexture(device: GfxDevice, num: number): DecodedTexture | null {
        if (this.decodedTextures[num] === undefined) {
            try {
                this.decodedTextures[num] = loadTextureFromTable(device, this.tex1Tab, this.tex1Bin, num);
            } catch (e) {
                console.warn(`Failed to load texture 0x${num.toString(16)} due to exception:`);
                console.error(e);
                this.decodedTextures[num] = makeFalseTexture(device, num);
            }
        }

        return this.decodedTextures[num];
    }
}