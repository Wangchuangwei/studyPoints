<template>
    <div>
        <!-- <el-button @click="analysis" style="margin: 0 0 20px 10px">开始全数据集分析</el-button>     -->
        <div class="show_result" v-if="evaluate_loaded">{{loaded_title}}</div>
        <div class="show_result" v-if="evaluate_analysis">{{analysis_title}}</div>
        <div class="title" v-if="codeData.length !== 0">Code差异分析结果</div>
        <el-table :data="codeData" v-if="codeData.length !== 0" style="width:70%" max-height="500px">
            <el-table-column prop="Add" label="ADD lines"/>
            <el-table-column prop="Del" label="DEL lines"/>
        </el-table>
        <div class="title" v-if="astData.length !== 0" style="margin-top: 20px">AST差异分析结果</div>
        <el-table :data="astData" v-if="astData.length !== 0" style="width:70%" max-height="500px">
            <el-table-column prop="Insert" label="Insert node"/>
            <el-table-column prop="Delete" label="Delete node"/>
            <el-table-column prop="Move" label="Move node"/>
            <el-table-column prop="Update" label="Update node"/>
        </el-table>
    </div>
</template>

<script setup lang="ts">
import {ref, onMounted} from 'vue'

// const evaluate_loaded = ref(false)
const evaluate_loaded = ref(true)
// let loaded_title = ref('数据集加载中...')
let loaded_title = ref('数据集加载完成')
// const evaluate_analysis = ref(false)
const evaluate_analysis = ref(true)
// let analysis_title = ref('analysis...')
let analysis_title = ref('analysis finished!')

let codeData = ref<any>([])
let astData = ref<any>([])

const initAnalysis = () => {
    evaluate_loaded.value = false
    evaluate_analysis.value =false
    loaded_title.value = '数据集加载中...'
    analysis_title.value = 'analysis...'
}

onMounted(() => {
    codeData.value = [
        {Add: 5, Del: 4}
    ]
    astData.value = [
        {Insert: 5, Delete: 4, Move: 3, Update: 3},
    ]
})

const analysis = () => {
    initAnalysis()
    evaluate_loaded.value = true

    codeData.value = [
        {Add: 15, Del: 29}
    ]
    astData.value = [
        {Insert: 15, Delete: 29, Move: 25, Update: 23},
    ]
}

</script>

<style scoped>

</style>