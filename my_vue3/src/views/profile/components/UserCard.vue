<template>
    <el-card style="margin-bottom: 20px;">
        <template #header>
            <span>About author</span>
        </template>
        <div class="user-profile">
            <div class="box-center">
                <def-pan-thumb >
                    <template #content>
                        <div class="informcoes">
                            <p>Hello</p>
                            <p>{{userStore.username}}</p>
                        </div>
                    </template>
                </def-pan-thumb>
            </div>
            <div class="box-center">
                <div class="user-name text-center">{{aboutAuthor.username}}</div>
                <div class="user-role text-center text-muted">Developer</div>
                <div class="user-role text-center text-muted">{{aboutAuthor.email}}</div>
            </div>
        </div>
        <div class="user-bio">
            <div class="user-bio-section">
                <div class="user-bio-section-header flex_center">
                    <def-svg-icon name="education" color="#303133"></def-svg-icon>
                    <span>Education</span>                
                </div>
                <div class="user-bio-section-body">
                    <div class="text-muted">{{aboutAuthor.education}}</div>
                </div>
            </div>
            <div class="user-bio-section">
                <div class="user-bio-section-header flex_center">
                    <def-svg-icon name="skill" color="#303133"></def-svg-icon>
                    <span>Skills</span>                
                </div>
                <div class="user-bio-section-body">
                    <div class="progress-item">
                        <span>Vue</span>
                        <el-progress :percentage="70" />
                    </div>
                    <div class="progress-item">
                        <span>TypeScript</span>
                        <el-progress :percentage="18" />
                    </div>
                    <div class="progress-item">
                        <span>Css</span>
                        <el-progress :percentage="12" />
                    </div>
                    <div class="progress-item">
                        <span>ESLint</span>
                        <el-progress :percentage="100" status="success" />
                    </div>    
                </div>
            </div>            
        </div>
    </el-card>
</template>

<script setup lang="ts" name="UserCard">
import { AboutAuthor } from '@/store/modules/setting'
import useUserStore from '@/store/modules/user'

const userStore = useUserStore()
const aboutAuthor = AboutAuthor()
</script>

<style scoped lang="scss">
.box-center {
  margin: 0 auto;
  display: table;
}
img {
    // display: inline-block;
    width: 100px;
    height: 100px;
    border-radius: 50%;
}
.user-profile {
    .user-name {
        font-weight: bold;
    }
    .box-center {
        padding-top: 10px;
    }
    .user-role {
        padding-top: 10px;
        font-weight: 400;
        font-size: 14px;
    }
    .informcoes {
        position: relative;
        top: 30%;
    }
}
.text-muted {
  color: #777;
}
.user-bio {
    margin-top: 15px;
    color: #606266;
    .user-bio-section {
        font-size: 14px;
        padding: 15px 0;
        .user-bio-section-header {
            border-bottom: 1px solid #dfe6ec;
            justify-content: flex-start;
            padding-bottom: 10px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        span {
            padding-left:10px;
        }
    }
}
</style>