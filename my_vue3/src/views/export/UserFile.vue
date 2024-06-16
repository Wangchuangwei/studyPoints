<template>
    <div>
        <div>
            <filename-option :filename="filename" @updateFilename="updateFilename"></filename-option>
            <book-type-option :bookType="bookType" @updateBookType="handelBookType"></book-type-option>
            <el-button :loading="downloadLoading" style="margin-bottom:20px" type="primary" icon="Document" @click="handleDownload">Export Selected Items</el-button>
        </div>        
    </div>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import BookTypeOption from './components/BookTypeOption.vue'
import FilenameOption from './components/FilenameOption.vue'

const bookType = ref('xlsx')
const filename =ref('')
const downloadLoading = ref(false)

const handelBookType = (type: string) => {
    bookType.value = type
}
const updateFilename = (name: string) => {
    console.log('filename:', name)
    filename.value = name
}
const handleDownload = () => {
    console.log('export')
}
</script>

<style scoped>

</style>