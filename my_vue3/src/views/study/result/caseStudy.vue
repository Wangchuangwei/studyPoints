<template>
    <div class="block-flex" style="margin-bottom: 20px">
            <div style="display: inline-block;">
            <div class="setting-title">选择实验的数据集</div>
            <el-radio-group v-model="radioSelect">
                <el-radio label="test">模型训练中数据集</el-radio>
                <el-radio label="dataset">
                    已有数据集
                    <el-select v-model="expDataset" placeholder="Select">
                        <el-option
                            v-for="item in options1"
                            :key="item.hash"
                            :label="item.name"
                            :value="item.hash"
                        />
                    </el-select>                    
                </el-radio>
                <el-radio label="define">
                    模型生成的提交信息
                    <el-select v-model="expDataset" placeholder="Select">
                        <el-option
                            v-for="item in options1"
                            :key="item.hash"
                            :label="item.name"
                            :value="item.hash"
                        />
                    </el-select>                    
                </el-radio>
                <el-radio label="upload">上传提交信息
                    <el-upload ref="uploadRef" class="upload-demo" drag multiple action="#" :auto-upload="false" :show-file-list="false" 
                        :on-change="handleChange" 
                    >
                        <!-- <el-icon class="el-icon--upload"><upload-filled style="width: 1em; height: 1em" /></el-icon> -->
                        <def-svg-icon class="icon--upload" name="code" width="30px" height="30px"></def-svg-icon>
                        <!-- <div class="el-upload__text">拖拽文件到这里或者<em>点击上传</em></div> -->
                    </el-upload>
                </el-radio>
            </el-radio-group>
        </div> 
    </div>
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
        <el-button @click="analysis" style="margin-left: 10px">开始全数据集分析</el-button>    
    </div>
    <div class="show_result" v-if="evaluate">模型加载完成！</div>
    <div class="show_result" v-if="evaluate">analysis finished!</div>
    <div class="title">预测结果</div>
    <el-table :data="testData" style="width:70%" max-height="500px">
        <el-table-column prop="Model" label="Model"/>
        <el-table-column prop="dataset" label="dataset"/>
        <el-table-column prop="ACC" label="ACC"/>
        <el-table-column prop="F1" label="F1"/>
        <el-table-column prop="AUC" label="AUC"/>
    </el-table>
    <!-- <div v-if="evaluate"><box-diagram></box-diagram></div> -->
</template>

<script setup lang="ts">
import { ref, reactive, watch, onMounted} from 'vue'
import type {exportFile, Records, ExportTableResponseData} from '@/api/type/index'
import {reqAllDatasetList} from '@/api/download'
import boxDiagram from '@/components/Diagram/box.vue'

defineProps({
    options: {
        type: Array<string>
    }
})
const value = ref('Mucha')
const radio = ref('generation')
let options = ref<Array<string>>(['CODISUM', 'CoreGen', 'CCRep', 'CodeBert', 'UniXcoder', 'Mucha', 'CLMF'])
const evaluate = ref(false)
const testNumber = ref(15)

const radioSelect = ref('test')

// watch(
//     () => radio.value, 
//     (newValue) => {
//         if (newValue == 'generation') options.value = ['CODISUM', 'CoreGen', 'CCRep', 'CodeBert', 'UniXcoder', 'Mucha']
//         else options.value = ['CodeBert', 'UniXcoder', 'Mucha']
//     }
// )

const analysis = () => {
    console.log('分析')
    evaluate.value = true
}

let options1 = ref<Records>([])
let expDataset = ref("")
onMounted(async() => {
    let res: ExportTableResponseData = await reqAllDatasetList()
    if (res.code === 200) {
        options1.value = res.data.records
        expDataset.value = options1.value[0].hash
    }
})

const testData = [
    {Model: 'CLMF', dataset: 'middle.tsv测试集',BLEU: 10.57, METEOR: 7.56, ROUGE: 8.06, ACC: 0.687, F1: 0.782, AUC: 0.708, MCC: 0.221},
    {Model: 'CodeBERT', dataset: 'val.tsv',BLEU: 3.42, METEOR: 4.51, ROUGE: 2.28, ACC: 0.605, F1: 0.714, AUC: 0.603, MCC: 0.156}
]
</script>

<style scoped>
.upload-demo {
    display: inline-block;
    vertical-align: middle;
}
</style>