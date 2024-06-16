<template>
    <div class="title" style="margin-top: 20px">已有数据集</div>
    <el-table :data="tableData" stripe class="table-demo" >
        <el-table-column prop="date" label="时间" sortable/>
        <el-table-column prop="name" label="名字" />
        <el-table-column prop="size" label="大小">
            <template #default="scope">
                {{formatFileSize(scope.row.size)}}
            </template>
        </el-table-column>
        <el-table-column label="操作">
            <template #default="scope">
                <div class="block-flex">
                    <el-button type="primary" @click="handleRemoveUploadFileList(scope.row)">删除</el-button>
                    <el-button type="primary" @click="handleView(scope.row)">预览</el-button>  
                </div>
            </template>
        </el-table-column>
    </el-table>
    <Viewer ref="fileViewerRef" :dialogTitle="dialogTitle"></Viewer>
</template>

<script setup lang="ts">
import {ref, reactive, onMounted} from 'vue'
import {reqAllDatasetList} from '@/api/download'
import type {exportFile, Records, ExportTableResponseData} from '@/api/type/index'
import { deleteByFileName, getFileContent } from '@/api/file'
import Viewer  from '@/components/Preview/index.vue'
import { ElNotification } from 'element-plus'

const fileViewerRef = ref<any>(null)
const dialogTitle = ref<string>("")
let tableData = ref<Records>([])

const emits = defineEmits(['updateDataset'])

const handleRemoveUploadFileList = async (row: exportFile) => {
    const index = tableData.value.findIndex((item: exportFile) => item.hash === row.hash)
    tableData.value.splice(index, 1)
    const result = await deleteByFileName({filename: row.name, fileHash: row.hash, type: 'dataset'})
    emits('updateDataset')
}

const handleView = async (row: exportFile) => {
    const result = await getFileContent('dataset', row.name, row.hash)
    const fileExtension = row.name.split('.').pop()
    dialogTitle.value = row.name
    if (fileExtension == 'xlsx') {
        fileViewerRef.value?.viewXlsx(new Blob([result]))
    }
    else if (fileExtension == 'docx') {
        fileViewerRef.value?.viewDocx(new Blob([result]))
    }
    else if (fileExtension == 'pdf') {
        fileViewerRef.value?.viewPdf(new Blob([result], { type: "application/pdf" }))
    } 
    else if (fileExtension == 'txt' || fileExtension == 'tsv') {
        fileViewerRef.value?.viewPdf(new Blob([result]))
    } 
    else if (fileExtension == 'json') {
        const reader = new FileReader()
        reader.readAsText(result)
        reader.onload = (e: any) => {
            fileViewerRef.value?.viewJson(e.target.result)  
        }
    }
    else {
        console.log('暂不支持')
        ElNotification({
            type: 'warning',
            message: '暂不支持查看该类型',
        })
    }
}

onMounted(() => {
    getDataset()
})

const getDataset = async () => {
    console.log('bef:', tableData.value)
    let res: ExportTableResponseData = await reqAllDatasetList()
    console.log('res11:', res.data.records)
    if (res.code === 200) {
        tableData.value = res.data.records
    }
}
const formatFileSize = (size: number) => {
    if (size === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(size) / Math.log(k))
    return parseFloat((size / Math.pow(k, i)).toFixed(2)) + ' ' +sizes[i]
}
 
defineExpose({
    getDataset
})
</script>

<style scoped>

</style>