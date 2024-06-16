<template>
    <div class="app-container">
        <el-radio-group v-model="uploadType" class="ml-4">
            <el-radio label="before" size="large">变更前</el-radio>
            <el-radio label="after" size="large">变更后</el-radio>
        </el-radio-group>
        <upload-file :uploadType="uploadType"></upload-file>
    </div>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import UploadFile from '@/components/UploadFile/index.vue'

const uploadType = ref('before')
</script>

<style scoped>

</style>