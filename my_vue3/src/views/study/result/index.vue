<template>
    <div>
        <el-tabs v-model="activeTab">
            <el-tab-pane label="提交生成" name="all">
                <keep-alive><full-analysis :options="options"></full-analysis></keep-alive>
            </el-tab-pane>
            <el-tab-pane label="评审预测" name="some">
                <keep-alive><case-study :options="options"></case-study></keep-alive>
            </el-tab-pane>
        </el-tabs>
    </div>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import fullAnalysis from './fullAnalysis.vue'
import caseStudy from './caseStudy.vue'

let activeTab = ref('all')

const options = ['CODISUM', 'CoreGen', 'CCRep', 'CodeBert', 'UniXcoder', 'Mucha']

</script>

<style lang="scss">
.el-tabs__active-bar {
    background-color: $study-theme;
}
.el-tabs__item.is-active  {
    color: $study-theme
}
.block-flex {
    display: flex;
    align-items: flex-end;
}
</style>
<style lang="scss" scoped>
</style>