<template>
    <div style="display:inline-block;">
        <label class="radio-label" style="padding-left:0;">Filename:</label>
        <el-input v-model="value" placeholder="Please enter the file name (default excel-list)" style="width:345px;" prefix-icon="Document" clearable />
    </div>
</template>

<script setup lang="ts">
import {computed} from 'vue'

const props = defineProps({
    filename: {
        type: String,
        default: ''
    }
})
const emits = defineEmits(['updateFilename'])
const value = computed({
    get: () => props.filename,
    set: (newValue) => {
        emits('updateFilename', newValue)
    }
})
</script>

<style scoped>

</style>