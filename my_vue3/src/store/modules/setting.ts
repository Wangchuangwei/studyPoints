import { defineStore } from "pinia";

let useLayOutSettingStore = defineStore('SettingStore', {
    state: () => {
        return {
            isCollapse: false,
            refsh: false,
        }
    }
})

let AboutAuthor = defineStore('',{
    state: () => {
        return {
            username: 'Chuangwei Wang',
            education: 'School of Computer Science and Technology, Soochow University',
            email: '1032273697@qq.com'
        }
    },
    actions: {
        authorInfo() {
            // console.log('hahs1111')
            this.username = 'haha',
            this.email = '123'
        }
    }
})

export default useLayOutSettingStore
export {AboutAuthor}