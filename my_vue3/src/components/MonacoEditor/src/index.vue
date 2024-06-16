<template>
    <div class="container">
        <el-row class="row-cols-2">
            <el-col :span="12" :xs="12" v-if="!isDiff">
                <label class="control-label">Language</label>
                <el-select v-model="language" class="form-select">
                    <el-option v-for="item in languages" :key="item.value" :label="item.label" :value="item.value"></el-option>
                </el-select>
            </el-col>
            <el-col :span="12" :xs="12">
                <label class="control-label">Theme</label>
                <el-select v-model="theme" class="form-select">
                    <el-option v-for="item in themes" :key="item.value" :label="item.label" :value="item.value"></el-option>
                </el-select>
            </el-col>
        </el-row>
        <editor :isDiff="isDiff" :oldValue="original" :value="value" :theme="theme"></editor>
    </div>
</template>

<script setup lang="ts">
import Editor from './Editor.vue'
import {ref} from 'vue'

const themes = [
    {
        value: 'vs',
        label: 'vs',
    },
    {
        value: 'hc-black',
        label: 'hc-black',
    },
    {
        value: 'vs-dark',
        label: 'vs-dark',
    }
]
const theme:string = ref('vs')
const languages = [
    {
        value: 'python',
        label: 'python',
    },
    {
        value: 'typescript',
        label: 'typescript',
    },
    {
        value: 'cpp',
        label: 'cpp',
    }
]
const language:string = ref('python')

defineProps({
    isDiff: Boolean,
    original: String,
    value: String
})

</script>

<style scoped lang="scss">
.container {
    padding: 0 16px;
    .row-cols-2 {
        margin: 10px 0;
        .control-label {
            color: #212529;
            font-weight: 400;
            display: inline-block;
        }
        .form-select {
            display: block;
            padding-top: 0.75rem;
            padding-bottom: 0.25rem;
        }
        .el-col:nth-child(1) {
            padding-right: 16px;
        }
    }    
}

</style>