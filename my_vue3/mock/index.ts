import {createUserList} from './user'
import {createProjectList, getChildData} from './project'
import {createDownloadList} from './download'

export default [
    // 用户登录接口
    {
        url: '/api/user/login',//请求地址
        method: 'post',//请求方式
        response: ({ body }) => {
            //获取请求体携带过来的用户名与密码
            const { username, password } = body;
            //调用获取用户信息函数,用于判断是否有此用户
            const checkUser = createUserList().find(
                (item) => item.username === username && item.password === password,
            )
            //没有用户返回失败信息
            if (!checkUser) {
                return { code: 201, data: { message: '账号或者密码不正确' } }
            }
            //如果有返回成功信息
            const { token } = checkUser
            return { code: 200, data: token  }
        },
    },
    // 获取用户信息
    {
        url: '/api/user/info',
        method: 'get',
        response: (request) => {
            //获取请求头携带token
            const token = request.headers.token;
            //查看用户信息是否包含有次token用户
            const checkUser = createUserList().find((item) => item.token === token)
            //没有返回失败的信息
            if (!checkUser) {
                return { code: 201, data: { message: '获取用户信息失败' } }
            }
            //如果有返回成功信息
            return { code: 200, data: checkUser }
        },
    },
    // 用户退出登录
    {
        url: '/api/user/logout',
        method: 'post',
        response: ({body}) => {
            //如果有返回成功信息
            return { code: 200, data: {message: '退出登录'} }
        },
    },
    // 文件根列表
    {
        url: '/api/project/rootData',
        method: 'get',
        response: (request) => {
            const token = request.headers.token;
            const rootData = createProjectList().find(item =>  item.token === token)
            if (!rootData) return {code: 201, message: '获取根数据失败'}
            return {code: 200, data: rootData}
        }
    }, 
    // 当前文件的子文件
    {
        url: '/api/project/childData',
        method: 'get',
        response: (request) => {
            const token = request.headers.token;
            const id = request.query.id
            const rootData = createProjectList().find(item =>  item.token === token)
            const ids = id.split('-')
            const childData = getChildData(rootData.project,'',ids)
            if (childData == undefined) return {code: 201, message: '获取根数据失败'}
            return {code: 200, data: childData}
        }
    },
    // 根据页码以及搜索字段获取导出的文件
    {
        url: '/api/download/studyExport/:page/:pageSize',
        method: 'get',
        response: (request) => {
            const token = request.headers.token;
        
            // 提取动态参数
            const urlParts = request.url.split('/api/download/studyExport/')[1].split('?')[0].split('/');
            const page = Number(urlParts[0])
            const pageSize = Number(urlParts[1])
            const {fileName} = request.query
            console.log('params:', page, pageSize, 'query:', fileName)
            const downloadList = createDownloadList()
            if (downloadList.length === 0) return {code: 200, data: { records: [], total: 0}}
            if (fileName) {
                const filterList = downloadList.filter(item =>item.name.includes(fileName))
                if (filterList.length === 0) return {code: 200, data: { records: [], total: 0}}
                const currentList = filterList.slice((page - 1) * pageSize, (page - 1) * pageSize + pageSize)

                return {code: 200, data: { records: currentList, total: filterList.length}}
            }
            const currentList = downloadList.slice((page - 1) * pageSize, (page - 1) * pageSize + pageSize)

            // const id = request.query.id
            // const rootData = createProjectList().find(item =>  item.token === token)
            // const ids = id.split('-')
            // const childData = getChildData(rootData.project,'',ids)
            // if (childData == undefined) return {code: 201, message: '获取根数据失败'}
            return {code: 200, data: { records: currentList, total: downloadList.length}}
        }
    },
]