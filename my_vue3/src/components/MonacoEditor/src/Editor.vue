<template>
    <div ref="container" class="app-monaco" :style="'height: ' + height + 'px'"></div>
</template>

<script setup lang="ts">
import assign from "nano-assign"
// import * as monaco from 'monaco-editor'
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api'
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";
import {
    ref,
    reactive,
    onMounted, 
    onBeforeUnmount,
    nextTick, 
    watch,
    toRaw
} from 'vue'

const props = defineProps({
    isDiff: {
        type: Boolean,
        default: true,
    },
    language: {
        type: String,
        default: "json",
    },
    oldValue: String,
    value: String,
    theme: String,
    height: {
        type: Number,
        default: 380,
    },
})
const defaultOpts = reactive({
  value: "",
  theme: "vs", // 编辑器主题：vs, hc-black, or vs-dark，更多选择详见官网
  roundedSelection: false, // 右侧不显示编辑器预览框
  autoIndent: true, // 自动缩进
  readOnly: true, // 是否只读
  diffWordWrap:true,
  wordWrap:'on',
  automaticLayout:true,
  scrollBeyondLastLine:false,
  scrollbar:{
    verticalScrollbarSize: 0
  },
})
const container = ref(null)

const se: any = self;
const tsArr = ["typescript", "javascript"];
const jsonArr = ["json"];

let editor: monaco.editor.IStandaloneCodeEditor | null = null
let diffEditor: monaco.editor.IStandaloneDiffEditor | null = null
let originalModel
let modifiedModel:  monaco.editor.ITextModel 

const initMonaco = () => {
    // console.log('高亮及提示')
    se.MonacoEnvironment = {
        getWorker(_: any, label: any) {
        if (jsonArr.includes(label)) {
            return new jsonWorker();
        }
        if (tsArr.includes(label)) {
            return new tsWorker();
        }
        return new editorWorker();
        },
    };

    console.log('初始化Monaco')
    const options:monaco.editor.IStandaloneEditorConstructionOptions = assign({
        value: props.value,
        them: props.theme,
        language: props.language,
        automaticLayout: true,
    }, defaultOpts)
    if(props.isDiff) {
        diffEditor = monaco.editor.createDiffEditor(container.value, options)
        originalModel = monaco.editor.createModel(
            props.oldValue!,
            // JSON.stringify(JSON.parse(props.oldValue), null, "t"),
            props.language
        )
        modifiedModel = monaco.editor.createModel(
            props.value!,
            // JSON.stringify(JSON.parse(props.value), null, "t"),
            props.language
        )
        console.log('aa')
        diffEditor.setModel({
            original: originalModel,
            modified: modifiedModel
        })
        console.log('bbb')
    } else {
        console.log('标准化')
        editor = monaco.editor.create(container.value, options)
    }
}

onMounted(() => {
    nextTick(() => {
        initMonaco()
    })
})

onBeforeUnmount(() => {
    editor && editor.dispose()
    diffEditor && diffEditor.dispose()
})

const getModifiedEditor: monaco.editor.IStandaloneCodeEditor | null = () => {
    if (props.isDiff) {
        if (diffEditor == null) return null
        return diffEditor.getModifiedEditor()
    } else {
        if (editor == null) return null
        return editor
    }
}

watch(
    () => props.oldValue,
    (newValue) => {
        originalModel = monaco.editor.createModel(newValue!, props.language)
        diffEditor?.setModel({
            original: originalModel,
            modified: modifiedModel
        })
    }
)

// watch(
//     () => props.value,
//     (newValue) => {
//         if ((props.isDiff && diffEditor) || (!props.isDiff && editor)) {
//             const modifiedEditor = getModifiedEditor()
//             if(!modifiedEditor) {
//                 console.log('null', props.isDiff)
//                 return
//             }
//             if (newValue !== modifiedEditor.getValue()) {
//                 modifiedEditor.setValue(newValue)
//             }
// //             toRaw(editor.value).getValue()  替换editor.value.getValue
//         }
//     }
// )

// watch(
//     () => props.theme,
//     (newValue) => {
//         if ((props.isDiff && diffEditor) || (!props.isDiff && editor)) {
//             const modifiedEditor = getModifiedEditor()
//             if(!modifiedEditor) {
//                 console.log('null', props.isDiff)
//                 return
//             }
//             monaco.editor.setTheme(newValue)
//         }
//     }
// )

// watch(
//     () => props.language,
//     (newValue) => {
//         const modifiedEditor = 
//     }
// )
</script>

<style scoped>

</style>