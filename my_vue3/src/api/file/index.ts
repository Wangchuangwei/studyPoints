import request from '@/utils/request'
import type {ResponseData, fileResponseData, uploadResponseData, ContentResponseData, fileData} from '../type/index'

enum API {
    ROOTDATA_URL = '/api/project/rootData',
    CHILDDATA_URL = '/api/project/childData',
    UPLOADFILE_URL = '/api/project/uploadFile',
    VERIFYUPLOAD = '/api/project/verifyUpload',
    MERGEUPLOAD = '/api/project/mergeUpload',
    DELETEBYFILENAME = '/api/project/deleteByFileName',
    GETFILECONTENT = '/api/project/getFileContent',
    GETFILELIST = '/api/project/getFileList',
    GETCHILDFILELIST = '/api/project/getChildFileList'
}

export const reqRootData = () => request.get<any, fileResponseData>(API.ROOTDATA_URL)

export const reqChildData = (id: string) => {
    const urlWithParams = `${API.CHILDDATA_URL}?id=${id}`
    return request.get<any, fileResponseData>(urlWithParams)
}

export const uploadFileData = (data: FormData, onUploadProgress: any) => {
    return request.post<any, ResponseData>(API.UPLOADFILE_URL, data, {headers: {'Content-Type': 'multipart/form-data'}, onUploadProgress: onUploadProgress})
}

export const verifyUpload = (filename: string, fileHash: string, type: string) => request.post<any, uploadResponseData>(API.VERIFYUPLOAD, JSON.stringify({filename, fileHash, type}))

export const mergeRequest = (data: any) => request.post(API.MERGEUPLOAD, JSON.stringify(data), {headers: {'Content-Type': 'application/json'}})

export const deleteByFileName = (data: any) => request.post<any, ResponseData>(API.DELETEBYFILENAME, JSON.stringify(data))

export const getFileContent = (type: string, filename: string, fileHash: string) => {
    const urlWithParams = `${API.GETFILECONTENT}?type=${type}&filename=${filename}&fileHash=${fileHash}`
    return request.get<any, any>(urlWithParams, {responseType: 'blob'})
}

export const getFileList = (type: string) => {
    const urlWithParams = `${API.GETFILELIST}?type=${type}`
    return request.get<any, fileResponseData>(urlWithParams)
}

export const getChildFileList = (type: string, id: string) => {
    const urlWithParams = `${API.GETCHILDFILELIST}?type=${type}&id=${id}`
    return request.get<any, fileResponseData>(urlWithParams)
}