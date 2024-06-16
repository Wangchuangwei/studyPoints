<template>
    <div class="app-container">
        <el-card style="margin-bottom: 10px">
            <filename-option :filename="keyword" @updateFilename="updateFilename"></filename-option>
            <el-button type="primary" style="margin-left: 20px;" @click="search">Search</el-button>
            <el-button style="margin-left: 20px;" @click="clearFilter">Reset all filters</el-button>
            <!-- <book-type-option :bookType="bookType" @updateBookType="handelBookType"></book-type-option> -->
            <el-button :loading="downloadLoading" style="margin-left: 20px;" type="primary" icon="Document" @click="handleDownload">Export Selected Items</el-button>
        </el-card>  

        <el-card>
            <el-table ref="tableRef" :data="tableData" @selection-change="handleSelectionChange" table-layout="auto" :default-sort="{prop: 'date', order: 'descending'}" style="margin: 10px 0;width：100%" border fit highlight-current-row>
                <el-table-column type="selection" align="center" width="50"/>
                <el-table-column prop="date" label="Date" sortable>
                    <template #default="scope">
                        <div style="display: flex; align-items: center">
                            <el-icon><timer /></el-icon>
                            <span style="margin-left: 10px">{{ scope.row.date }}</span>
                        </div>
                    </template>
                </el-table-column>
                <el-table-column prop="name" label="Filename" />
                <el-table-column prop="tag" label="Tag"
                    :filters="[
                        {text: 'Message', value: 'Message'},
                        {text: 'Prediction', value: 'Prediction'}
                    ]"
                    :filter-method="filterTag"
                    filter-placement="bottom-end"
                >
                    <template #default="scope">
                        <el-tag :type="scope.row.tag === 'Message' ? '':'success'" disable-transitions>{{scope.row.tag}}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column label="Operaions">
                    <template #default="scope">
                        <el-button size="small" icon="Bottom"  @click="handleExport(scope.$index, scope.row)">Export</el-button>
                        <el-button size="small" icon="View" @click="handlePreview(scope.$index, scope.row)">Preview</el-button>
                        <el-popconfirm :title="`确定要删除${scope.row.name}?`" width="260px" @confirm="handleDelete(scope.$index, scope.row)">
                            <template #reference>
                                <el-button size="small" type="danger" icon="Delete">Delete</el-button>
                            </template>
                        </el-popconfirm>
                    </template>
                </el-table-column>
            </el-table>  
            <el-pagination v-model:current-page="pageNo" v-model:page-size="pageSize" :total="total" :page-sizes="[3, 5, 7, 9]" :background="true" layout="prev, pager, next, jumper, ->, sizes, total,"
                @current-change="getHasFile" 
                @size-change="sizeHandler" 
            />
        </el-card>    
        <Viewer ref="fileViewerRef"></Viewer>
    </div>
</template>

<script setup lang="ts">
import {ref, computed, onMounted} from 'vue'
import JSZip from 'jszip'
import {saveAs} from 'file-saver'
import {reqAllFileList} from '@/api/download'
import type {exportFile, Records, ExportTableResponseData} from '@/api/type/index'
import BookTypeOption from './components/BookTypeOption.vue'
import FilenameOption from './components/FilenameOption.vue'
import Viewer  from '@/components/Preview/index.vue'
import { getFileContent } from '@/api/file'

const tableRef = ref()
const bookType = ref('xlsx')
const keyword =ref('')
const downloadLoading = ref(false)
const fileViewerRef = ref<any>(null)

let pageNo = ref<number>(1)
let pageSize = ref<number>(3)
let total = ref<number>(0)

let tableData = ref<Records>([])
let multipleSelection = ref<Records>([])

onMounted(() => {
    getHasFile()
})

/**
 * 搜索文件
 */
const search = () => {
    if (keyword.value.length == 0) return
    getHasFile()
}

const handelBookType = (type: string) => {
    bookType.value = type
}
const updateFilename = (name: string) => {
    keyword.value = name
}
/**
 * 导出所选择的文件
 */
const handleDownload = () => {
    console.log('export')
    if (multipleSelection.value.length) {
        downloadLoading.value = true
        const files = [
            {name: '修改.docx', label: '测试用例文档_修改.docx', fileHash: '21c7403041006058bc694761729f8cca.docx'},
            {name: '修改.txt', label: '测试中文.txt', fileHash: '4af184db012c7d1bf98133ea11ba7536.txt'},
            {name: '修改.pdf', label: '测设英文.pdf', fileHash: 'de8f1847d37fa82d4c3adca8b2048382.pdf'},
            {name: '修改1.docx', label: 'Highlights.docx', fileHash: 'ad9902a4236d4466a651c83a6e913bbd.docx'},
        ]
        const zip = new JSZip()
        const folder = zip.folder('下载文件')
        const promises = files.map(file => {
            return getFileContent('before', file.label, file.fileHash).then(data => {
                const name = file.name
                folder!.file(name, data, {binary: true})
            })
        })
        Promise.all(promises).then(() => {
            zip.generateAsync({type: 'blob'}).then(content => {
                saveAs(content, '下载啦')
            })
        })
    }
}
const handleSelectionChange = (val: Records) => {
    multipleSelection.value = val
}

const clearFilter = () => {
    keyword.value = ''
    getHasFile()
    tableRef.value!.clearFilter()
}
const filterTag = (value: string, row: exportFile) => row.tag === value

/**
 * 针对当前行自定义的操作
 */
const handleExport = (index: number, row: exportFile) => {
    console.log('row:', index, row)
    // // 下载文件 
    // const result = await getFileContent(type, data.label, data.fileHash)
    // let a = document.createElement('a')
    // a.href = URL.createObjectURL(result)
    // a.download = "出事.docx"
    // a.style.display = "none"
    // document.body.appendChild(a)
    // a.click()
    // a.remove()
}
const handleDelete = (index: number, row: exportFile) => {
    console.log('row:', index, row)
    // const result = await deleteByFileName({filename: file.name, fileHash: hash, type: uploadType.value})
    // console.log('result:', result)
}
const handlePreview = (index: number, row: exportFile) => {
    console.log('row:', index, row)
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

/**
 * 根据页码动态获取当前页面下的table数据
 */
const getHasFile = async (pager = 1) => {
    pageNo.value = pager
    let res: ExportTableResponseData = await reqAllFileList(
        pageNo.value,
        pageSize.value,
        keyword.value
    )
    if (res.code === 200) {
        total.value = res.data.total
        tableData.value = res.data.records
    }
}
const sizeHandler = () => {
    getHasFile()
}
</script>

<style>
.radio-label {
  font-size: 14px;
  color: #606266;
  line-height: 40px;
  padding: 0 12px 0 30px;
}
</style>