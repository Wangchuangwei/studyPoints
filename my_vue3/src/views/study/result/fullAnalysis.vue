<template>
    <div class="block-flex" style="margin-bottom: 20px">
        <!-- <div style="display: inline-block">
            <div class="setting-title">模型类型</div>
            <el-radio-group v-model="radio">
                <el-radio label="generation">提交生成</el-radio>
                <el-radio label="prediction">评审预测</el-radio>
            </el-radio-group>
        </div> -->
        <div style="display: inline-block;">
            <div class="setting-title">选择模型</div>
            <el-select v-model="value" placeholder="Select">
                <el-option
                    v-for="item in options"
                    :key="item"
                    :label="item"
                    :value="item"
                />
            </el-select>
        </div> 
        <div style="display: inline-block; margin-left: 10px">
            <div class="setting-title">选择实验的数据集</div>
            <el-select v-model="expDataset" placeholder="Select">
                <el-option
                    v-for="item in options1"
                    :key="item.hash"
                    :label="item.name"
                    :value="item.hash"
                />
            </el-select>
        </div>
        <el-button @click="analysis" style="margin-left: 10px">开始全数据集分析</el-button>    
    </div>
    <!-- <div class="show_result" v-if="evaluate">Epoch模型加载完成！</div> -->
    <div class="show_result" v-if="evaluate">Epoch模型加载完成！</div>
    <div class="show_result" v-if="evaluate">analysis finished!</div>
    <div class="title">模型性能结果</div>
    <el-table v-if="radio == 'generation'" :data="testData" style="width:70%" max-height="500px">
        <el-table-column prop="Model" label="Model"/>
        <el-table-column prop="dataset" label="dataset"/>
        <el-table-column prop="BLEU" label="BLEU"/>
        <el-table-column prop="METEOR" label="METEOR"/>
        <el-table-column prop="ROUGE" label="ROUGE-L"/>
    </el-table>
    <el-table v-else :data="testData" style="width:70%" max-height="500px">
        <el-table-column prop="Model" label="Model"/>
        <el-table-column prop="ACC" label="ACC"/>
        <el-table-column prop="F1" label="F1"/>
        <el-table-column prop="AUC" label="AUC"/>
    </el-table>
    <div class="download_model" style="margin-top: 10px">
        <div><span>下载</span>当前模型生成的提交信息</div>
        <el-button @click="downloadModel">下载文件</el-button>
    </div>
    <div class="show_result" style="margin-bottom: 0" v-if="download">当前模型生成的提交信息下载完成！</div>
    <div v-if="radio == 'prediction' && evaluate"><box-diagram></box-diagram></div>
</template> 

<script setup lang="ts">
import { ref, reactive, watch, onMounted} from 'vue'
import type {exportFile, Records, ExportTableResponseData} from '@/api/type/index'
import {reqAllDatasetList} from '@/api/download'
import boxDiagram from '@/components/Diagram/box.vue'
import request from '@/utils/request'
import axios from 'axios'

defineProps({ 
    options: {
        type: Array<string>
    }
})

const radio = ref('generation')
const value = ref('Mucha')
let options = ref<Array<string>>(['CODISUM', 'CoreGen', 'CCRep', 'CodeBert', 'UniXcoder', 'Mucha'])
let expDataset = ref("")

const evaluate = ref(false)

let options1 = ref<Records>([])
// watch(
//     () => radio.value, 
//     (newValue) => {
//         if (newValue == 'generation') options.value = ['CODISUM', 'CoreGen', 'CCRep', 'CodeBert', 'UniXcoder', 'Mucha']
//         else options.value = ['CodeBert', 'UniXcoder', 'Mucha']
//     }
// )

onMounted(async() => {
    let res: ExportTableResponseData = await reqAllDatasetList()
    if (res.code === 200) {
        options1.value = res.data.records
        expDataset.value = options1.value[0].hash
    }
})

const download = ref(false)
const downloadModel = () => {
    // if (trainData.value.length === 0) return
    download.value = true
}

const testData = [
    {Model: 'Mucha', dataset: 'val.tsv',BLEU: 10.57, METEOR: 7.56, ROUGE: 8.06, ACC: 80.56, F1: 0.872, AUC: 0.731},
    {Model: 'CCRep', dataset: 'val.tsv',BLEU: 3.42, METEOR: 4.51, ROUGE: 2.28, ACC: 80.56, F1: 0.872, AUC: 0.731}
]

const analysis = () => {
    console.log('分析a')
    // request.get<any, any>('/remote/login?fileName=分析')
    // request.get<any, any>('/expe/submit_task?username=a&project_name=b&task_name=cg')
    // axios.get('/remote/login', {
    //     params: {
    //         fileName: 'analysis'
    //     }
    // });

    evaluate.value = true
    
}
</script>

<style scoped lang="scss">
.download_model {
    display: flex;
    width: 800px;
    justify-content: space-between;
    align-items: center;
    padding-right: 40px;
    margin-bottom: 20px;
    span {
        // margin-left: 10px;
        font-weight: 600;
    }
}
</style>