import { DataFetcher } from '../DataFetcher';

import { GameInfo } from './scenes';

function sliceDataView(data: DataView, byteOffset: number, byteLength?: number): DataView {
    return new DataView(data.buffer, data.byteOffset + byteOffset, byteLength);
}

class SFAObject {
    public name: string;
    public objClass: number;

    constructor(public objType: number, data: DataView) {
        this.name = '';
        this.objClass = data.getInt16(0x50);
        let offs = 0x91;
        let c;
        while ((c = data.getUint8(offs)) != 0) {
            this.name += String.fromCharCode(c);
            offs++;
        }
    }
}

export class ObjectManager {
    private objectsTab: DataView;
    private objectsBin: DataView;
    private objindexBin: DataView;

    constructor(private gameInfo: GameInfo) {
    }

    public async create(dataFetcher: DataFetcher) {
        const pathBase = this.gameInfo.pathBase;
        this.objectsTab = (await dataFetcher.fetchData(`${pathBase}/OBJECTS.tab`)).createDataView();
        this.objectsBin = (await dataFetcher.fetchData(`${pathBase}/OBJECTS.bin`)).createDataView();
        this.objindexBin = (await dataFetcher.fetchData(`${pathBase}/OBJINDEX.bin`)).createDataView();
    }

    public loadObject(objType: number): SFAObject {
        objType = this.objindexBin.getUint16(objType * 2);
        const offs = this.objectsTab.getUint32(objType * 4);
        return new SFAObject(objType, sliceDataView(this.objectsBin, offs));
    }
}