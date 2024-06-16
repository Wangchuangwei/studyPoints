<template>
    <div class="title">数据集分析</div>
    <div class="block-flex" style="margin-bottom: 15px">
        <div style="display: inline-block;">
            <div class="setting-title">选择分析的数据集</div>
            <el-select v-model="dataset" placeholder="Select">
                <el-option
                    v-for="item in options"
                    :key="item.hash"
                    :label="item.name"
                    :value="item.hash"
                />
            </el-select>
        </div> 
        <el-button @click="analysis" style="margin-left: 10px">开始全数据集分析</el-button>    
    </div>
    <div class="show_result" v-if="evaluate_loaded">{{loaded_title}}</div>
    <div class="show_result" v-if="evaluate_analysis">{{analysis_title}}</div>
    <div class="title" v-if="allData.length !== 0">数据集分析结果</div>
    <el-table :data="allData" style="width:90%" max-height="500px" v-if="allData.length !== 0">
        <!-- <el-table-column prop="task" label="任务"/> -->
        <el-table-column prop="name" label="文件名"/>
        <el-table-column prop="train" label="训练集"/>
        <el-table-column prop="val" label="验证集"/>
        <el-table-column prop="test" label="测试集"/>
        <el-table-column prop="all" label="总共"/>
    </el-table>
    <div class="title" style="margin-top: 20px">选择实验数据集</div>
    <div class="block-flex" style="margin-bottom: 15px">
        <div style="display: inline-block">
            <div class="setting-title">模型类型</div>
            <el-radio-group v-model="radio">
                <el-radio label="generation">提交生成</el-radio>
                <el-radio label="prediction">代码评审</el-radio>
            </el-radio-group>
        </div>
        <div style="display: inline-block; margin-left: 10px">
            <div class="setting-title">选择实验的数据集</div>
            <el-select v-model="expDataset" placeholder="Select">
                <el-option
                    v-for="item in options"
                    :key="item.hash"
                    :label="item.name"
                    :value="item.hash"
                />
            </el-select>
        </div> 
        <el-button @click="configDataset" style="margin-left: 10px">配置</el-button>   
    </div>
    <div class="show_result" >当前<strong>提交生成任务</strong>实验数据集文件：{{cur_ExpDataset_gene}}</div>
    <div class="show_result" >当前<strong>代码评审任务</strong>实验数据集文件：{{cur_ExpDataset_pred}}</div>
</template>

<script setup lang="ts">
import {ref, onMounted, watch, nextTick} from 'vue'
import {useGeneDataSettingStore, usePredDataSettingStore} from '@/store/modules/dataset'
import {reqAllDatasetList} from '@/api/download'
import type {exportFile, Records, ExportTableResponseData} from '@/api/type/index'

let useGeneDataStore = useGeneDataSettingStore()
let usePredDataStore = usePredDataSettingStore()
let dataset = ref("")
let expDataset = ref("")
let options = ref<Records>([])
let cur_ExpDataset_gene = ref("")
let cur_ExpDataset_pred = ref("")

const evaluate_loaded = ref(true)
let loaded_title = ref('数据集加载完成！')
const evaluate_analysis = ref(true)
let analysis_title = ref('analysis finished!')

let allData = ref<any>([{'task': '代码评审预测', 'name': 'middle.tsv','train': 13232, 'val': 1654, 'test': 1654, 'all': 16540}])
const radio = ref('generation')

const ws = new WebSocket('ws://10.10.65.82:8080/echo')

onMounted(() => {
    ws.addEventListener('open', handleWsOpen.bind(this), false)
    ws.addEventListener('close', handleWsClose.bind(this), false)
    ws.addEventListener('error', handleWsError.bind(this), false)
    ws.addEventListener('message', handleWsMessage.bind(this), false)
    getDatasetList()  
})

const getDatasetList = async () => {
    let res: ExportTableResponseData = await reqAllDatasetList()
    if (res.code === 200) {
        options.value = res.data.records
        dataset.value = options.value[0].hash
        expDataset.value = options.value[0].hash
        const hasGeneData = options.value.some((item: exportFile) => item.hash == useGeneDataStore.$state.hash) 
        const hasPredData = options.value.some((item: exportFile) => item.hash == usePredDataStore.$state.hash) 
        if (useGeneDataStore.$state.hash && hasGeneData) {
            cur_ExpDataset_gene.value = useGeneDataStore.$state.name
        }else {
            cur_ExpDataset_gene.value = '无'
        }
        if (usePredDataStore.$state.hash && hasPredData) {
            cur_ExpDataset_pred.value = usePredDataStore.$state.name
        }else {
            cur_ExpDataset_pred.value = '无'
        }
    }
}

const initAnalysis = () => {
    evaluate_loaded.value = false
    evaluate_analysis.value =false
    loaded_title.value = '数据集加载中...'
    analysis_title.value = 'analysis...'
}
const analysis = () => {
    initAnalysis()
    evaluate_loaded.value = true
    ws.send(JSON.stringify({type: 'datasetAnalysis'}))
    // allData.valus.push({'task': '代码评审预测', 'name': 'middle.tsv','train': 13232, 'val': 1654, 'test': 1654, 'all': 16540})
}
const handleWsOpen = () => {
    console.log('WebSocket2已经打开 ')
}
const handleWsClose = (e: any) => {
    console.log('WebSocket2关闭')
    console.log(e)
}
const handleWsError = (e:any) => {
    console.log('WebSocket2发生错误')
    console.log(e)
}
const handleWsMessage = (e:any) => {
    const result = JSON.parse(e.data)
    if (result['status'] == 'loaded') {
        loaded_title.value = '数据集加载完成！'
        evaluate_analysis.value = true
    }
    if (result['status'] == 'finished') {
        analysis_title.value = 'analysis finished!'
        allData.value.push(result['data'])
    }
}

const configDataset = () => {
    const filterDataset = options.value.filter((item: exportFile) => item.hash == expDataset.value)
    if (radio.value == 'generation') {
        useGeneDataStore.datasetInfo(filterDataset[0].name, expDataset.value)
        cur_ExpDataset_gene.value = filterDataset[0].name
    } else {
        usePredDataStore.datasetInfo(filterDataset[0].name, expDataset.value)
        cur_ExpDataset_pred.value = filterDataset[0].name
    }
}

defineExpose({
    getDatasetList
})
</script>

<style scoped>

</style>