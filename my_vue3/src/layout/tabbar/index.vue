<template>
    <el-header style="text-align: right; font-size: 12px; overflow: hidden;">
        <div class="toolbar">
            <div class="toolbar-left"><BreadCrumb /></div>
            <div class="toolbar-right"><Setting /></div>
        </div>
    </el-header>
</template>

<script setup lang="ts">
import BreadCrumb from './breadcrumb/index.vue'
import Setting from './setting/index.vue'
</script>

<style scoped lang="scss">
.toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    height: 100%;

    .toolbar-left {
        display: flex;
        align-items: center;
        margin-left: 10px;
    }

    .toolbar-right {
        display: flex;
        align-items: center;
    }
}
</style>