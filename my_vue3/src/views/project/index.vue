<template>
    <el-container class="adaptive-padding">
        <el-aside>
            <file-tree @file-click="handleFileClick"></file-tree>
        </el-aside>
        <el-main>
            文件面板
                <!-- <monaco-editor  :isDiff="true" :original="original" :value="newValue"></monaco-editor> -->

            <!-- <keep-alive>
                <monaco-editor  :isDiff="false" :original="original" :value="newValue"></monaco-editor>
            </keep-alive> -->
        </el-main>
        <Viewer ref="fileViewerRef"></Viewer>
    </el-container>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import FileTree from '@/components/FileTree/src/index.vue'
// import MonacoEditor from '@/components/MonacoEditor/src/index.vue'
import Viewer  from '@/components/Preview/index.vue'
import type { fileData, ContentResponseData } from '@/api/type/index'
import { getFileContent} from '@/api/file/index'

let newValue = ref("aasfdasdf")
let original = ref('aaa') 

const fileViewerRef = ref<any>(null)

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

<style scoped lang="scss">
.el-aside {
    width: $base-menu-width;
    transition: all 0.3s;
}
.el-main {
    // background: yellowgreen;
    border-left: 1px solid #e2e2e2
}
</style>