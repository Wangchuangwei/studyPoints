<template>
    <div class="down-tree">
        <el-tree :props="treeProps" :load="loadNode" lazy @node-click="handleNodeClick">
            <template #default="{node, data}">
                <span v-if="!data.isLeaf" class="node-folder">
                    <el-icon v-if="node.expanded" style="margin: 0 6px 0 0;" size="16"><FolderOpened/></el-icon>
                    <el-icon v-else style="margin: 0 6px 0 0;" size="16"><Folder/></el-icon>
                    <small :title="node.label">{{node.label}}</small>
                </span>
                <span v-else class="node-file">
                    <el-icon style="margin: 0 6px 0 0;" size="16"><Document/></el-icon>
                    <small :title="node.label">{{node.label}}</small>
                </span>
            </template>
        </el-tree>
    </div>
</template>

<script setup lang="ts">
import {ref} from 'vue'
import type Node from 'element-plus/es/components/tree/src/model/node'
import {reqRootData, reqChildData, getFileList, getChildFileList} from '@/api/file/index'
import {fileData, fileResponseData} from '@/api/type/index' 
import { ElNotification } from 'element-plus'

const treeProps ={
    label: 'label',
    children: 'children',
    isLeaf: 'isLeaf'
}

// show-checkbox 
const props = defineProps({
    fileType: {
        type:String,
        default: 'before'
    }
})

const emits = defineEmits(['node-click'])

const loadNode = async (node: Node, resolve: (data: fileData[]) => void) => {
    try {
        if (node.level === 0) {
            // const resData: fileResponseData = await reqRootData()
            // resolve(resData.data.project as fileData[])
            // console.log('type fill:', props.fileType)
            const resData: fileResponseData = await getFileList(props.fileType)
            // console.log('resData:', resData.data)
            resolve(resData.data as fileData[])
        } else {
            setTimeout(async() => {
                // const resData = await reqChildData(node.data.id)
                const resData = await getChildFileList(props.fileType, node.data.id)
                if (resData.code === 201) {
                    return Promise.reject(new Error(resData.message))
                }
                resolve(resData.data)
            }, 1000)
        }
    } catch (error) {
        ElNotification({
            type: 'error',
            message: (error as Error).message
        })
    }
}

const handleNodeClick = (data: fileData) => {
    // console.log('data:', data, props.fileType)
    if (!data.isLeaf) return 
    emits('node-click', props.fileType, data)
}
</script>

<style lang="scss" scoped>
.down-tree {
    height: inherit;
    &::v-deep {
        .el-tree-node.is-expanded > .el-tree-node__children {
            display: inline;
            min-width: 100% !important;
        }
    }
} 
.el-tree-node {
    .el-tree-node__content {
        .node-file {
            display: flex;
            align-items: center;
            small {
                font-weight: normal;
                color: #40485c;
                transition: all ease 0.3s;
            }
        }
        .node-folder {
            display: flex;
            align-items: center;
            small {
                font-weight: bold;
                color: #40485c;
                transition: all ease 0.3s;
            }
        }
    }
}
</style>