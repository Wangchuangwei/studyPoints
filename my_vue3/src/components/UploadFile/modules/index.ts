import SparkMD5 from 'spark-md5'

const SIZE = 1024 * 20 // 200kb

/**
 * 文件切片
 */
 const createFileChunk = (file: any, size = SIZE) => {
    const fileChunkList = []
    let cur = 0
    while (cur < file.size) {
        const end = Math.min(file.size, cur + size)
        fileChunkList.push({
            file: file.raw?.slice(cur, end)
        })
        cur += size
    }
    return fileChunkList
}

/**
 * 文件内容hash
 */
 const calculateHash = (file: any) => {
    return new Promise((resolve, reject) => {
        const spark = new SparkMD5.ArrayBuffer()
        const fileReader = new FileReader()
        fileReader.readAsArrayBuffer(file.raw)
        fileReader.onload = (e: any) => {
            spark.append(e.target.result)
            resolve(spark.end())
        }
        fileReader.onerror = () => {
            reject('')
        }
    })
}

export {createFileChunk, calculateHash}