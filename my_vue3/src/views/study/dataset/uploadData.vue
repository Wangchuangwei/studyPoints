<template>
    <el-card>
        <div class="title">上传数据集</div>
        <el-upload ref="uploadRef" class="upload-demo" drag multiple action="#" :auto-upload="false" :show-file-list="false" 
            :on-change="handleChange" 
        >
            <!-- <el-icon class="el-icon--upload"><upload-filled style="width: 1em; height: 1em" /></el-icon> -->
            <def-svg-icon class="icon--upload" name="code" width="30px" height="30px"></def-svg-icon>
            <div class="el-upload__text">拖拽文件到这里或者<em>点击上传</em></div>
        </el-upload>
        <el-table :data="tableData" stripe class="table-demo" v-if="visibleTable">
            <el-table-column prop="name" label="名字" />
            <el-table-column prop="size" label="大小">
                <template #default="scope">
                    {{formatFileSize(scope.row.size)}}
                </template>
            </el-table-column>
            <el-table-column prop="precentage" label="进度">
                <template #default="scope">
                    <el-progress :text-inside="true" :stroke-width="26" :percentage="scope.row.percentage"></el-progress>
                </template>
            </el-table-column>
            <el-table-column prop="percentage" label="操作">
                <template #default="scope">
                    <!-- <div style="display: flex; flex-wrap: nowarp"> -->
                        <el-button type="primary" @click="handleRemoveUploadFileList(scope.row)">删除</el-button>
                        <el-button type="primary" v-if="scope.row.status=='fail'" @click="handleResume(scope.row)">继续</el-button>   
                        <!-- <el-button type="primary" v-else-if="scope.row.status=='success'" @click="handleView(scope.row)">预览</el-button>                            -->
                    <!-- </div> -->
                </template>
            </el-table-column>
        </el-table>
    </el-card>
</template>

<script setup lang="ts">
import {ref, reactive, nextTick} from 'vue'
import {UploadProps, UploadFile, UploadFiles} from 'element-plus'
import {uploadFileData, verifyUpload, mergeRequest, deleteByFileName, getFileContent, getFileList} from '@/api/file/index'
import type {uploadData, ContentResponseData} from '@/api/type/index'
import {createFileChunk, calculateHash} from '@/components/UploadFile/modules/index.ts'

// const uploadType = ref('before')
const SIZE = 1024 * 20 // 200kb
const tableData = reactive<UploadFiles>([])
const requestLists = ref<any>([])
const fileViewerRef = ref<any>(null)
const visibleTable = ref(false)

const props = defineProps({
    uploadType: {
        type: String,
        default: 'dataset'
    }
})
const emits = defineEmits(['updateDataset'])

/**
 * el-upload内置的change函数，文件上传或者上传成功时的回调，不过这里因为
 * :auto-upload="false"的缘故，上床成功的回调不会执行
 * @param uploadFile el-upload当前上传的文件对象
 * @param uploadFiles el-upload上传的文件列表
 */
const handleChange: UploadProps['onChange'] = async (uploadFile, uploadFiles) => {
    visibleTable.value = true
    // uploadFile.pause = true
    uploadFile.status = 'uploading'
    console.log('file:', uploadFile.size, uploadFile)
    tableData.push({...uploadFile})
    const index = tableData.findIndex(item => item.uid === uploadFile.uid)
    // console.log('tableData:', tableData)
    uploadFilehandle(uploadFile, index, props.uploadType)
}

/**
 * 文件分片和hash合并
 */
const uploadFilehandle = async(file: any, tabDataIndex: number, type: string) => {
    requestLists.value = []
    const fileChunkList = createFileChunk(file)
    const hash = await calculateHash(file) as string
    const {shouldUpload, uploadedList} = await verifyUpload(file.name, hash, type)
    console.log('shouldUpload:', shouldUpload, 'uploadedList:', uploadedList)

    if (!shouldUpload) {
        tableData[tabDataIndex].percentage = 100
        tableData[tabDataIndex].status = 'success'
        await nextTick()
        return 
    }
    let loaded = 0

    let data: uploadData[] = fileChunkList.map(({file}, index) => ({
        fileHash: hash,
        index,
        hash: `${hash}-${index}`,
        chunk: file,
        size: file.size,
        percentage: 0,
    })).filter(({hash, size}) => {
        if (uploadedList.includes(hash)) {
            loaded += size
            return false
        }
        return true
    })
    const percentage = Number((loaded * 100 / file.size).toFixed(2))
    uploadChunks(type, data, tabDataIndex, file.name, hash, file.size, percentage)
}

/**
 * 分块文件上传至服务器
 */
 const uploadChunks = async (type: string, data: uploadData[], tabDataIndex: number, filename: string, filehash: string, size: number, basePercentage: number) => {
    const requestList = data.map(({chunk, hash}) => {
        const form = new FormData()
        form.append('type', type)
        form.append('chunk', chunk)
        form.append('hash', hash)
        form.append('filename', filename)
        form.append('fileHash', filehash)
        return form
    }).map(async (form, index) => uploadFileData(form, onUploadProgress(data, index, tabDataIndex, size, basePercentage)))
    try {
        requestLists.value.push(requestList)
        const results = await Promise.all(requestList)
        const success = results.filter(item => item.code === 200)
        if (success.length === data.length){
            await mergeRequest({type: type, size: SIZE, fileHash: filehash, filename: filename}) 
            tableData[tabDataIndex].percentage = 100
            tableData[tabDataIndex].status = 'success'
            emits('updateDataset')
            console.log('全部上传')
        } else {
            console.log(filename, '切片没有完全上传')
        }
    } catch (error) {
        tableData[tabDataIndex].status = 'fail'
    }
}

/**
 * 上传进度条
 */
const onUploadProgress = (data: uploadData[], index: number, tabDataIndex: number, size: number, basePercentage: number) => {
    return (event: any) => {
        data[index].percentage = event.loaded / event.total
        const loaded = data.filter(item => item.percentage == 1).reduce((acc, cur) => acc + cur.size, 0)
        tableData[tabDataIndex].percentage = Number((loaded * 100 / size + basePercentage).toFixed(2)) 
    }
}
 
/**
 * 自定义的表格方法
 */
const handleRemoveUploadFileList = async (file: UploadFile) => {
    const index = tableData.findIndex((item: UploadFile) => item.uid === file.uid) 
    const hash = await calculateHash(file) as string
    if (index !== -1) {
        tableData.splice(index, 1)
        const result = await deleteByFileName({filename: file.name, fileHash: hash, type: props.uploadType})
        console.log('result:', result)
        emits('updateDataset')
        if (tableData.length === 0) {
            visibleTable.value = false
        }
    }
}

const handleResume = async (file: UploadFile) => {
    file.status = 'uploading'
    const index = tableData.findIndex(item => item.uid === file.uid)
    uploadFilehandle(file, index, props.uploadType)
}

const formatFileSize = (size: number) => {
    if (size === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(size) / Math.log(k))
    return parseFloat((size / Math.pow(k, i)).toFixed(2)) + ' ' +sizes[i]
}
 
</script>

<style scoped lang="scss">
.upload-demo {
    // width: 300px;
    margin: 0 auto;
    .el-upload__text {
        display: inline-block;
    }
}
.table-demo {
    // width: 300px;
    margin: 0 auto;
}

.icon--upload {
    vertical-align: middle;
    margin-right: 10px;
}  
</style>
<style lang="scss">
.el-upload-dragger {
    padding: 20px ;
    background-color: #f0f2f6;
} 
</style>