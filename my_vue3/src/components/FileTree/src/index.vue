<template>
<el-scrollbar>
    <el-tabs v-model="activeTab" type="card" class="project-tabs">
        <el-tab-pane v-for="(tab, index) in tabs" :key="index" :label="tab.label" :name="tab.name">
            <keep-alive>
                <tree v-if="activeTab === tab.name" :fileType="activeTab"  @node-click="handleNodeClick"></tree>
            </keep-alive>
        </el-tab-pane>
    </el-tabs>
</el-scrollbar>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import Tree from './Tree.vue'
import { TabsPaneContext } from 'element-plus'
import { fileData } from '@/api/type/index'

const activeTab = ref('before')
const tabs = [
    {label: '变更前', name: 'before'},
    {label: '变更后', name: 'after'}
]

const emits = defineEmits(['file-click'])

const handleNodeClick = (type: string, data: fileData) =>{
    emits('file-click', type, data)
}
</script>

<style scoped lang="scss">
.project-tabs {
    width: 100%;
    height: 100%;
    .custom-tabs-label{
        display: grid;
        grid-template-columns: 1fr 1fr;
    }    
}
</style>