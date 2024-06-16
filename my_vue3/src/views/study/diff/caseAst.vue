<template>
    <div style="margin-bottom: 20px">
        <div style="display: inline-block; margin-left: 10px">
            <div class="setting-title">输入数据编号</div>
            <el-input v-model="astNumber" type="number"/>
        </div> 
        <el-button @click="analysis" style="margin-left: 10px">开始个例分析</el-button> 
    </div>
    <div class="show_result" v-if="evaluate_loaded">{{loaded_title}}</div>
    <div class="show_result" v-if="evaluate_analysis">{{analysis_title}}</div>
    <div v-if="evaluate">
        <div class="title">Code差异对比分析结果</div> 
        <div class="block-flex" style="margin-bottom: 10px">
            <div style="display: inline-block; margin-left: 10px">
                <div class="setting-title">Code对比</div>
                <el-button @click="getCodeDiff" >查看Code对比结果</el-button> 
            </div>        
        </div>
        <el-table :data="codeData" style="width:70%" max-height="500px">
            <el-table-column prop="Add" label="ADD"/>
            <el-table-column prop="Del" label="DEL"/>
        </el-table>
        <div class="title" style="margin-top: 20px">AST解析和对比分析结果</div> 
        <div class="block-flex" style="margin-bottom: 10px">
            <div style="display: inline-block; margin-left: 10px">
                <div class="setting-title">AST解析</div>
                <el-radio-group v-model="radio">
                    <el-radio label="before">变更前</el-radio>
                    <el-radio label="after">变更后</el-radio>
                </el-radio-group>
            </div>
            <el-button @click="getAst" style="margin-left: 20px">查看AST解析结果</el-button>          
        </div>
        <div class="block-flex" style="margin-bottom: 10px">
            <div style="display: inline-block; margin-left: 10px">
                <div class="setting-title">AST对比</div>
                <el-button @click="getAstDiff" >查看AST对比结果</el-button> 
            </div>        
        </div>
        <el-table :data="astData" style="width:70%" max-height="500px">
            <el-table-column prop="Insert" label="Insert node"/>
            <el-table-column prop="Delete" label="Delete node"/>
            <el-table-column prop="Move" label="Move node"/>
            <el-table-column prop="Update" label="Update node"/>
        </el-table>
    </div>

    <Viewer ref="fileViewerRef"></Viewer>
</template>

<script setup lang="ts">
import { ref, reactive, watch} from 'vue'
import Viewer  from '@/components/Preview/index.vue'
import { getFileContent } from '@/api/file'

const evaluate_loaded = ref(false)
let loaded_title = ref('数据加载中...')
const evaluate_analysis = ref(false)
let analysis_title = ref('差异对比...')
const evaluate = ref(false)

let codeData = ref<any>([])
let astData = ref<any>([])

const astNumber = ref(15)
const radio = ref('before')
const fileViewerRef = ref<any>(null)

const initAnalysis = () => {
    evaluate_loaded.value = false
    evaluate_analysis.value =false
    loaded_title.value = '数据获取成功'
    analysis_title.value = '差异对比完成'
}

const analysis = () => {
    initAnalysis()
    evaluate_loaded.value = true

    codeData.value = [
        {Add: 2, Del: 2}
    ]
    astData.value = [
        {Insert: 3, Delete: 3, Move: 6, Update: 1},
    ]

    setTimeout(() => {
        evaluate_analysis.value =true
        evaluate.value = true
    }, 1000)
}

const getAst = async () => {
    const result = await getFileContent(radio.value, '测试字符.json', '574f1a87d5bd35d2ac6b80110caf2e32')
    const reader = new FileReader()
    reader.readAsText(result)
    reader.onload = (e: any) => {
        fileViewerRef.value?.viewJson(e.target.result)  
    }
}

const getAstDiff = async () => {
    console.log('ast diff')
    const result = await getFileContent(radio.value, "对比.txt", "c23fc5c0690cdc52a0208aaf9a95eeaf")
    fileViewerRef.value?.viewPdf(new Blob([result]))
}

const getCodeDiff = async () => {
    console.log('code diff')
}

const handleFileClick = async (type: string, data: fileData) => {
    console.log('fileClick:', type, data)
    // const result = await getFileContent(type, data.label, data.fileHash)
    // // xlsx
    // fileViewerRef.value?.viewXlsx(new Blob([result]))
    // // docx
    // fileViewerRef.value?.viewDocx(new Blob([result]))
    // // pdf
    // fileViewerRef.value?.viewPdf(new Blob([result], { type: "application/pdf" }))
    // // text
    // fileViewerRef.value?.viewPdf(new Blob([result]))
}
</script>

<style scoped>
.block-flex {
    display: flex;
    align-items: flex-end;
}
</style>