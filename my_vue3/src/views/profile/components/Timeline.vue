<template>
    <el-timeline>
        <el-timeline-item v-for="(item, index) in timeline" :key="index" :timestamp="item.timestamp" placement="top">
            <el-card>
                <h2 class="title">{{item.title}}</h2>
                <p class="content">{{item.content}}</p>
            </el-card>
        </el-timeline-item>
    </el-timeline>
</template>

<script setup lang="ts" name="Timeline">

interface timeLine {
    timestamp: string,
    title: string,
    content: string
}

const timeline: Array<timeLine> = [
        {
          timestamp: '2023/12/20',
          title: 'Update Main Menu template',
          content: 'ChuangweiWang committed 2023/12/20 18:31'
        },
        {
          timestamp: '2023/11/15',
          title: 'Update Github template',
          content: 'ChuangweiWang committed 2023/11/15 17:11'
        },
        {
          timestamp: '2023/11/2',
          title: 'Build Template',
          content: 'ChuangweiWang committed 2023/11/2 15:23'
        },
        {
          timestamp: '2023/10/23',
          title: 'Release New Version',
          content: 'ChuangweiWang committed 2023/10/23 20:46'
        }
      ]

</script>

<style scoped lang="scss">
.title {
    font-size: 14px;
    font-weight: 900;
    margin-bottom: 20px;
}
.content {
    font-size: 14px;
}
</style>