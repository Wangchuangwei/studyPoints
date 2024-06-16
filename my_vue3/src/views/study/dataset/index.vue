<template>
    <div>
        <el-container>
            <el-aside style="width: 40%;">
                <upload-data @updateDataset="updateDataset"></upload-data>
                <loaded-data @updateDataset="updateDataset" ref="refLoadedData"></loaded-data>
            </el-aside>
            <el-main>
                <keep-alive>
                    <analysis-data ref="refAnalysisData"></analysis-data>
                </keep-alive>
            </el-main>
        </el-container>
        
    </div>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import uploadData from './uploadData.vue'
import loadedData from './loadedData.vue'
import analysisData from './analysisData.vue'

const refLoadedData = ref<any>(null)
const refAnalysisData = ref<any>(null)

const updateDataset = () => {
    refLoadedData.value?.getDataset() 
    refAnalysisData.value?.getDatasetList()
}
</script>

<style scoped>
</style>
<style lang="scss">
.block-flex {
    display: flex;
    align-items: flex-end;
}
</style>