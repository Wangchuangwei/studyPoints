import { defineStore } from "pinia";

let useGeneDataSettingStore = defineStore('SettingDataset', {
    state: () => {
        return {
            name: "",
            hash: ""
        }
    }, 
    actions: {
        datasetInfo(name: string, hash: string) {
            this.name = name
            this.hash = hash
        },
    }
})

let usePredDataSettingStore = defineStore('SettingDataset', {
    state: () => {
        return {
            name: "",
            hash: ""
        }
    }, 
    actions: {
        datasetInfo(name: string, hash: string) {
            this.name = name
            this.hash = hash
        },
    }
})

// export default useGeneDataSettingStore
export {useGeneDataSettingStore, usePredDataSettingStore}