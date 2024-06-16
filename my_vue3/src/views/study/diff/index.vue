<template>
    <div>
        <el-tabs v-model="activeTab">
            <el-tab-pane label="全集分析" name="all">
                <keep-alive><all-ast></all-ast></keep-alive>
            </el-tab-pane>
            <el-tab-pane label="个例分析" name="some">
                <keep-alive><case-ast></case-ast></keep-alive>
            </el-tab-pane>
        </el-tabs>
    </div>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import allAst from "./allAst.vue";
import caseAst from "./caseAst.vue";

let activeTab = ref('all')

</script>

<style scoped>

</style>