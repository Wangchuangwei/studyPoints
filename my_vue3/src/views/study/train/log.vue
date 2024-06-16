<template>
    <div style="margin-top: 20px">
        <div class="title">实时日志</div>
        <div class="log-list" v-html="logContent"></div>    
    </div>
</template>

<script setup lang="ts">
import {ref, onMounted} from 'vue'
const logContent = ref('')

const updateLog = (result: any) => {
    console.log(result)
    // logContent.value += result.log + '<br>'
}

onMounted(()=> {
    logContent.value = 'Loading datasets <br> ' + 
    'The size of after process train set: 13232 <br> ' +
    'The size of after process val set: 1654 <br> ' +
    'The size of after process test set: 1654 <br> ' +
    'Datasets loaded successfully <br> Building the model <br> Hidden_size:768 <br>' + 
    'Trainable parameters: 322M <br>' +
    'Model built successfully <br> Start straining <br>' + 
    'Num examples = 16540 <br>' +
    'Batch size = 8 <br>' +
    'Num epoch = 12  <br>' + '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'
    + '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'
    + '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'
    + '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'
    + '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'
    + '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'
    + '10 <br>'+ '10 <br>'+ '10 <br>'+ '10 <br>'
})

defineExpose({
    updateLog
})
</script>

<style scoped lang="scss">
.log-list {
    background-color: #f0f2f6;
    width: 90%;
    height: 200px;
    padding: 10px;
    overflow: auto;
}
</style>