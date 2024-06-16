<template>
    <div style="display:inline-block;">
        <label class="radio-label">Book Type:</label>
        <el-select v-model="value" style="width:120px">
            <el-option v-for="(item, index) in options" :key="index" :label="item" :value="item" />
        </el-select>
    </div>
</template>

<script setup lang="ts">
import {computed} from 'vue'

const props = defineProps({
    bookType: {
        type: String,
        default: 'xlsx'
    }
})
const emits = defineEmits(['updateBookType'])

const options = ['csv', 'xlsx', 'txt']
const value = computed({
    get: () => props.bookType,
    set: (newValue) => {
        if (newValue !== props.bookType) {
            emits('updateBookType', newValue)
        }
    }
}) 
</script>

<style scoped lang="scss">
</style>