/**
 * 响应是否成功
 */
export interface ResponseData {
    code?: number,
    message?: string,
    ok?: boolean
}

export interface LoginFormData {
    username?: string,
    password?: string
}

/**
 * 文件树结构
 */
export interface fileData {
    id: string,
    filename?: string,
    fileHash?: string,
    label?: string,
    isLeaf?: boolean,
    children?: fileData[]
}

/**
 * 上传的每个文件切片数据格式
 */
export interface uploadData {
    fileHash: string,
    index: number,
    hash: string,
    chunk: any,
    size: number,
    percentage: number
}

/**
 * 用于显示文件table的每一行类型
 */
export interface exportFile {
    date: string,
    name: string,
    tag: string,
    hash?: string, 
    size?: string,
}

export type Records = exportFile[]

export interface ExportTableResponseData extends ResponseData {
    data: {
        records: Records,
        total: number,
        size?: number,
        current?: number,
        orders?: []
        optimizeCountSql?: boolean
        hitCount?: boolean
        countId?: null
        maxLimit?: null
        searchCount?: number
        pages?: number
    }
}

/**
 * 用户登录的响应数据
 */
export interface LoginResponseData extends ResponseData {
    data?: string
}

/**
 * 用户路由权限
 */
export interface userInfoResponseData extends ResponseData {
    data: {
        routes: string[]
        buttons: string[]
        roles: string[]
        username: string
        avatar: string
    }
}

/**
 * 返回的文件树
 * @param data 文件树结构内容
 */
export interface fileResponseData extends ResponseData {
    data: fileData[]
}

/**
 * 文件是否上传或者已上传的切片
 */
export interface uploadResponseData extends ResponseData {
    shouldUpload: boolean
    uploadedList: string[]
}

/**
 * 通用返回数据内容
 * @param data 内容
 */
export interface ContentResponseData extends ResponseData {
    data: string
}
