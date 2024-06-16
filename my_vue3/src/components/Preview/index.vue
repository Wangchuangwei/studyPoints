<template>
    <!-- docx -->
    <el-dialog
        v-model="dialogDocxValue"
        :title="dialogTitle"
        width="60%"
        @close="dialogDocxClose"
    >
        <div ref="docxRef" class="word-div"></div>
    </el-dialog>
    <!-- pdf/text -->
    <el-dialog
        v-model="dialogPdfValue"
        :title="dialogTitle"
        width="60%"
        @close="dialogPdfClose"
    >
        <div class="pdf-div">
            <iframe :src="iframeUrl" style="width: 100%; height:99%"></iframe>
        </div>
    </el-dialog>
    <!-- xlsx -->
    <el-dialog
        v-model="dialogXlsxValue"
        :title="dialogTitle"
        width="60%"
        @close="dialogXlsxClose"
    >
        <div class="xlsx-div">
            <el-tabs v-model="activeName" type="border-card">
                <el-tab-pane
                    v-for="(item, index) in excelSheet"
                    :key="index"
                    :label="item.name"
                    :name="item.name"
                >
                    <div class="table" v-html="item.html"></div>
                </el-tab-pane>
            </el-tabs>            
        </div>

    </el-dialog>
    <!-- json -->
    <el-dialog
        v-model="dialogJsonValue"
        :title="dialogTitle"
        width="60%"
        @close="dialogJsonClose"
    >
        <div class="json-div">
            <!-- <json-viewer :data="jsonData" :replace-tabs="true"></json-viewer> -->
            <pre v-html="jsonData"></pre>
        </div>
    </el-dialog>
</template>

<script setup lang="ts">
import {ref, reactive, nextTick} from 'vue'
// import JsonViewer from 'vue-json-pretty'
import {renderAsync} from 'docx-preview'
import * as XLSX from 'xlsx'

// import viewXlsx from '@/components/Preview/viewXlsx.vue'

const props = defineProps({
    dialogTitle: {
        type: String,
        default: '预览'
    }
})

/**
 * docx文件预览
 */
const docxRef = ref<any>()
const dialogDocxValue = ref(false)

const viewDocx = (data: any) => {
    docxRef.value = ""
    dialogDocxValue.value = true
    nextTick(() => {
        dialogDocxValue.value = true;
        renderAsync(data, docxRef.value)
    })
}
const dialogDocxClose = () => {
    dialogDocxValue.value = false
    docxRef.value = ""
}

/**
 * pdf和text文件预览
 */
const iframeUrl = ref("")
const dialogPdfValue = ref(false)

const viewPdf = (data: any) => {
    iframeUrl.value = ""
    dialogPdfValue.value = true
    nextTick(() => {
        dialogPdfValue.value = true
        iframeUrl.value = URL.createObjectURL(data)
    })
}
const dialogPdfClose = () => {
    dialogPdfValue.value = false
    iframeUrl.value = ""
}

/**
 * xlsx文件预览
 */
const excelSheet: any = ref([])
const activeName = ref("")
const dialogXlsxValue = ref(false)

const viewXlsx = (data: any) => {
    dialogXlsxValue.value = true
    const reader = new FileReader()
    reader.readAsArrayBuffer(data)
    reader.onload = (event: any) => {
        const arrayBuffer = event.target["result"]
        const workbook = XLSX.read(new Uint8Array(arrayBuffer), {
            type: "array"
        })
        const list = [];
        const sheetNames = workbook.SheetNames;
        activeName.value = sheetNames[0];
        for (const p of sheetNames) {
            let html = "";
            try {
                html = XLSX.utils.sheet_to_html(workbook.Sheets[p]);
            } catch (e) {
                html = "";
            }
            const map = {
                name: p,
                html: html
            };
            list.push(map);
        }
        excelSheet.value = list;
    }
}
const dialogXlsxClose = () => {
    dialogXlsxValue.value = false
    excelSheet.value = ""
    activeName.value = ""
}

/**
 * json文件预览
 */
let jsonData = reactive<any>({
    "root": {
        "type" : "15",
        "typeLable": "CompilationUnit",
        "pos": "0",
        "length": "101",
        "children": [
            {
                "type" : "55",
                "label": "class",
                "typeLable": "TypeDeclaration",
                "pos": "0",
                "length": "77",
                "children": []
            },
            {
                "type" : "83",
                "label": "public",
                "typeLable": "Modifier",
                "pos": "0",
                "length": "6",
                "children": []
            }
        ]
    }
})
const dialogJsonValue = ref(false)
const viewJson = (data: any) => {
    dialogJsonValue.value = false
    nextTick(() => {
        dialogJsonValue.value = true
        jsonData = syntaxHighlight(data)
    })
}
const dialogJsonClose = () => {
    dialogJsonValue.value = false
    jsonData = {}
}

const syntaxHighlight = (json: any) => {
    if (typeof json != 'string') {
        json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&').replace(/</g, '<').replace(/>/g, '>');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, (match: any) => {
        let cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
            cls = 'key';
            } else {
            cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

defineExpose({
    viewDocx,
    viewPdf,
    viewXlsx,
    viewJson
})
</script>

<style scoped>
.word-div {
    height: calc(70vh);
    overflow: auto;
}
.pdf-div {
    height: calc(70vh);
    overflow: auto;
}
.xlsx-div {
    height: calc(70vh);
    overflow: auto;
}
.json-div {
    height: calc(70vh);
    overflow: auto;
}
</style>
<style lang="scss">
.xlsx-div {
  .table-html-wrap table {
    border-right: 1px solid #fff;
    border-bottom: 1px solid #e8eaec;
    border-collapse: collapse;
    // margin: auto;
  }

  .table-html-wrap table td {
    border-left: 1px solid #e8eaec;
    border-top: 1px solid #e8eaec;
    white-space: wrap;
    text-align: left;
    min-width: 100px;
    padding: 4px;
  }

  table {
    border-top: 1px solid #ebeef5;
    border-left: 1px solid #ebeef5;
    width: 100%;
    // overflow: auto;

    tr {
      height: 44px;
    }

    td {
      min-width: 200px;
      max-width: 400px;
      padding: 4px 8px;
      border-right: 1px solid #ebeef5;
      border-bottom: 1px solid #ebeef5;
    }
  }

  .el-tabs--border-card > .el-tabs__content {
    overflow-x: auto;
  }
}
</style>