class ImageCompressor {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.debounceTimer = null;
        this.debounceDelay = 500; // 500ms的延迟
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.formatSelect = document.getElementById('formatSelect');
        this.qualitySlider = document.getElementById('qualitySlider');
        this.qualityValue = document.getElementById('qualityValue');
        this.originalImage = document.getElementById('originalImage');
        this.compressedImage = document.getElementById('compressedImage');
        this.originalSize = document.getElementById('originalSize');
        this.compressedSize = document.getElementById('compressedSize');
        this.compressionRatio = document.getElementById('compressionRatio');
        this.psnrValue = document.getElementById('psnrValue');
        this.msssimValue = document.getElementById('msssimValue');
        this.deleteButton = document.getElementById('deleteButton');
        this.wittControls = document.getElementById('wittControls');
        this.channelSelect = document.getElementById('channelSelect');
        this.snrSelect = document.getElementById('snrSelect');
    }

    bindEvents() {
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = '#666';
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.style.borderColor = '#ccc';
        });
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = '#ccc';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.processImage(file);
            }
        });
        this.fileInput.addEventListener('change', () => {
            const file = this.fileInput.files[0];
            if (file) {
                this.processImage(file);
            }
        });
        this.formatSelect.addEventListener('change', () => {
            this.wittControls.style.display = 
                this.formatSelect.value === 'witt' ? 'block' : 'none';
            const qualityControl = document.querySelector('.quality-control');
            qualityControl.style.display = 
                this.formatSelect.value === 'witt' ? 'none' : 'block';
            this.updateCompression();
        });
        this.qualitySlider.addEventListener('input', () => {
            this.qualityValue.textContent = this.qualitySlider.value;
        });
        this.qualitySlider.addEventListener('change', () => {
            this.updateCompression();
        });
        this.deleteButton.addEventListener('click', () => this.deleteImage());
        this.channelSelect.addEventListener('change', () => this.updateCompression());
        this.snrSelect.addEventListener('change', () => this.updateCompression());
    }

    async processImage(file) {
        try {
            const actualSize = file.size;
            this.originalSize.textContent = this.formatFileSize(actualSize);
            
            const reader = new FileReader();
            await new Promise((resolve, reject) => {
                reader.onload = resolve;
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
            
            this.originalImage.src = reader.result;
            await new Promise(resolve => this.originalImage.onload = resolve);
            
            this.deleteButton.style.display = 'block';
            await this.updateCompression();
        } catch (error) {
            console.error('图片处理错误:', error);
            alert('图片处理失败: ' + error.message);
        }
    }

    async updateCompression() {
        const format = this.formatSelect.value;
        const quality = parseInt(this.qualitySlider.value);
        
        try {
            if (format === 'bpg') {
                await this.compressWithBPG(quality);
            } else if (format === 'jpeg') {
                await this.compressWithJPEG(quality);
            } else if (format === 'witt') {
                await this.compressWithWITT(quality);
            }
        } catch (error) {
            console.error('压缩失败:', error);
            this.compressedImage.src = '';
            this.compressedSize.textContent = '-';
            this.compressionRatio.textContent = '-';
            this.psnrValue.textContent = '-';
            this.msssimValue.textContent = '-';
            alert(`${format.toUpperCase()} 压缩失败\n原因: ${error.message}`);
        }
    }

    async compressWithBPG(quality) {
        try {
            // 将0-100的质量值映射到0-51的范围
            const bpgQuality = Math.round(quality * 51 / 100);
            console.log('Mapped quality:', quality, 'to BPG quality:', bpgQuality);
            
            // 创建canvas获取图像数据
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // 确保原图已加载
            if (!this.originalImage.complete) {
                await new Promise(resolve => this.originalImage.onload = resolve);
            }
            
            canvas.width = this.originalImage.naturalWidth;
            canvas.height = this.originalImage.naturalHeight;
            
            // 确保尺寸有效
            if (canvas.width === 0 || canvas.height === 0) {
                throw new Error('Invalid image dimensions');
            }
            
            ctx.drawImage(this.originalImage, 0, 0);
            
            // 使用 Promise 包装 toBlob 调用
            const blob = await new Promise((resolve, reject) => {
                try {
                    canvas.toBlob(blob => {
                        if (blob) {
                            resolve(blob);
                        } else {
                            reject(new Error('Failed to create blob'));
                        }
                    }, 'image/png');
                } catch (error) {
                    reject(error);
                }
            });

            console.log('Blob created successfully, size:', blob.size);
            
            // 创建FormData对象
            const formData = new FormData();
            formData.append('image', blob, 'image.png');
            
            // 修改请求URL中的质量参数
            const response = await fetch(`http://localhost:8080/compress?quality=${bpgQuality}&format=bpg`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`服务器错误 ${response.status}: ${errorText}`);
            }
            
            console.log('Server response received');
            const result = await response.json();
            
            // 使用前端显示的原始大小
            const originalSizeBytes = this.getFileSizeInBytes(this.originalSize.textContent);
            const compressedSizeBytes = result.compressedSize;
            
            // 计算压缩率
            const compressionRatio = (compressedSizeBytes / originalSizeBytes) * 100;
            
            // 更新压缩图像和结果
            this.compressedImage.src = `data:image/png;base64,${result.imageData}`;
            this.compressedSize.textContent = this.formatFileSize(compressedSizeBytes);
            this.compressionRatio.textContent = compressionRatio.toFixed(3) + '%';
            this.psnrValue.textContent = result.psnr.toFixed(2) + ' dB';
            this.msssimValue.textContent = result.msssim.toFixed(4);
            
        } catch (error) {
            console.error('BPG压缩错误:', error);
            alert(`BPG压缩错误: ${error.message}`);
            throw error;
        }
    }

    async compressWithJPEG(quality) {
        try {
            // 确保JPEG质量在1-100范围内
            const jpegQuality = Math.max(1, Math.min(100, quality));
            console.log('JPEG quality:', jpegQuality);
            
            // 创建canvas获取图像数据
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // 确保原图已加载
            if (!this.originalImage.complete) {
                await new Promise(resolve => this.originalImage.onload = resolve);
            }
            
            canvas.width = this.originalImage.naturalWidth;
            canvas.height = this.originalImage.naturalHeight;
            
            ctx.drawImage(this.originalImage, 0, 0);
            
            // 将canvas转为blob
            const blob = await new Promise((resolve, reject) => {
                try {
                    canvas.toBlob(blob => {
                        if (blob) {
                            resolve(blob);
                        } else {
                            reject(new Error('Failed to create blob'));
                        }
                    }, 'image/png');
                } catch (error) {
                    reject(error);
                }
            });

            console.log('Blob created successfully, size:', blob.size);
            
            // 创建FormData对象
            const formData = new FormData();
            formData.append('image', blob, 'image.png');
            
            // 发送到服务器进行压缩
            const response = await fetch(`http://localhost:8080/compress?quality=${jpegQuality}&format=jpeg`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`服务器错误 ${response.status}: ${errorText}`);
            }
            
            const result = await response.json();
            
            // 使用前端显示的原始大小
            const originalSizeBytes = this.getFileSizeInBytes(this.originalSize.textContent);
            const compressedSizeBytes = result.compressedSize;
            
            // 计算压缩率，与BPG使用相同的计算方式
            const compressionRatio = (compressedSizeBytes / originalSizeBytes) * 100;
            
            // 更新压缩图像和结果
            this.compressedImage.src = `data:image/jpeg;base64,${result.imageData}`;
            this.compressedSize.textContent = this.formatFileSize(compressedSizeBytes);
            this.compressionRatio.textContent = compressionRatio.toFixed(3) + '%';
            this.psnrValue.textContent = result.psnr.toFixed(2) + ' dB';
            this.msssimValue.textContent = result.msssim.toFixed(4);
            
        } catch (error) {
            console.error('JPEG压缩错误:', error);
            alert(`JPEG压缩错误: ${error.message}`);
            throw error;
        }
    }

    async compressWithWITT(quality) {
        try {
            // 创建canvas并裁剪为256x256
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 256;
            canvas.height = 256;
            
            // 计算缩放和裁剪参数
            const scale = Math.max(canvas.width / this.originalImage.naturalWidth, 
                                 canvas.height / this.originalImage.naturalHeight);
            const scaledWidth = this.originalImage.naturalWidth * scale;
            const scaledHeight = this.originalImage.naturalHeight * scale;
            const x = (canvas.width - scaledWidth) / 2;
            const y = (canvas.height - scaledHeight) / 2;
            
            // 绘制并裁剪图片
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(this.originalImage, x, y, scaledWidth, scaledHeight);
            
            // 更新原图显示
            this.originalImage.src = canvas.toDataURL('image/png');
            
            // 获取裁剪后的图片大小
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/png');
            });
            this.originalSize.textContent = this.formatFileSize(blob.size);
            
            const formData = new FormData();
            formData.append('image', blob, 'image.png');
            
            const channel = this.channelSelect.value;
            const snr = this.snrSelect.value;
            
            const response = await fetch(
                `http://localhost:8080/compress?quality=${quality}&format=witt&snr=${snr}&channel=${channel}`, 
                {
                    method: 'POST',
                    body: formData
                }
            );
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`服务器错误 ${response.status}: ${errorText}`);
            }
            
            const result = await response.json();
            
            // 更新压缩图像和结果
            this.compressedImage.src = `data:image/png;base64,${result.imageData}`;
            this.compressedSize.textContent = this.formatFileSize(result.compressedSize);
            this.compressionRatio.textContent = result.compressionRatio.toFixed(3) + '%';
            this.psnrValue.textContent = result.psnr.toFixed(2) + ' dB';
            this.msssimValue.textContent = result.msssim.toFixed(4);
            
        } catch (error) {
            console.error('WITT压缩错误:', error);
            alert(`WITT压缩错误: ${error.message}`);
            throw error;
        }
    }

    updateCompressionResults(compressedData) {
        this.compressedImage.src = `data:image/png;base64,${compressedData.imageData}`;
        this.compressedSize.textContent = this.formatFileSize(compressedData.compressedSize);
        this.compressionRatio.textContent = compressedData.compressionRatio.toFixed(3) + '%';
        this.psnrValue.textContent = compressedData.psnr.toFixed(2) + ' dB';
        this.msssimValue.textContent = compressedData.msssim.toFixed(4);
    }

    async calculateImageQuality() {
        const psnr = await this.calculatePSNR(this.originalImage, this.compressedImage);
        const msssim = await this.calculateMSSSIM(this.originalImage, this.compressedImage);
        
        this.psnrValue.textContent = psnr.toFixed(2) + ' dB';
        this.msssimValue.textContent = msssim.toFixed(4);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        const size = (bytes / Math.pow(k, i)).toFixed(2);
        return size + ' ' + sizes[i];
    }

    async calculatePSNR(original, compressed) {
        return new Promise(async (resolve) => {
            try {
                if (!original.complete || !compressed.complete) {
                    await Promise.all([
                        new Promise(resolve => original.onload = resolve),
                        new Promise(resolve => compressed.onload = resolve)
                    ]);
                }

                const canvas1 = document.createElement('canvas');
                const canvas2 = document.createElement('canvas');
                const ctx1 = canvas1.getContext('2d');
                const ctx2 = canvas2.getContext('2d');
                
                canvas1.width = original.naturalWidth;
                canvas1.height = original.naturalHeight;
                canvas2.width = original.naturalWidth;
                canvas2.height = original.naturalHeight;
                
                ctx1.drawImage(original, 0, 0);
                ctx2.drawImage(compressed, 0, 0);
                
                const imageData1 = ctx1.getImageData(0, 0, canvas1.width, canvas1.height);
                const imageData2 = ctx2.getImageData(0, 0, canvas2.width, canvas2.height);
                
                let mse = 0;
                const data1 = imageData1.data;
                const data2 = imageData2.data;
                
                for (let i = 0; i < data1.length; i += 4) {
                    mse += Math.pow(data1[i] - data2[i], 2);
                    mse += Math.pow(data1[i+1] - data2[i+1], 2);
                    mse += Math.pow(data1[i+2] - data2[i+2], 2);
                }
                
                mse /= (canvas1.width * canvas1.height * 3);
                
                const psnr = mse === 0 ? Infinity : 20 * Math.log10(255) - 10 * Math.log10(mse);
                
                resolve(isFinite(psnr) ? psnr : 100);
            } catch (error) {
                console.error('PSNR calculation error:', error);
                resolve(0);
            }
        });
    }

    async calculateMSSSIM(original, compressed) {
        return new Promise(async (resolve) => {
            try {
                if (!original.complete || !compressed.complete) {
                    await Promise.all([
                        new Promise(resolve => original.onload = resolve),
                        new Promise(resolve => compressed.onload = resolve)
                    ]);
                }

                const canvas1 = document.createElement('canvas');
                const canvas2 = document.createElement('canvas');
                const ctx1 = canvas1.getContext('2d');
                const ctx2 = canvas2.getContext('2d');
                
                canvas1.width = original.naturalWidth;
                canvas1.height = original.naturalHeight;
                canvas2.width = original.naturalWidth;
                canvas2.height = original.naturalHeight;
                
                ctx1.drawImage(original, 0, 0);
                ctx2.drawImage(compressed, 0, 0);
                
                const imageData1 = ctx1.getImageData(0, 0, canvas1.width, canvas1.height);
                const imageData2 = ctx2.getImageData(0, 0, canvas2.width, canvas2.height);
                
                let ssimR = 0, ssimG = 0, ssimB = 0;
                const data1 = imageData1.data;
                const data2 = imageData2.data;
                const C1 = Math.pow(0.01 * 255, 2);
                const C2 = Math.pow(0.03 * 255, 2);
                
                for (let i = 0; i < data1.length; i += 4) {
                    const r1 = data1[i];
                    const r2 = data2[i];
                    ssimR += this.calculateSSIMForChannel(r1, r2, C1, C2);
                    
                    const g1 = data1[i + 1];
                    const g2 = data2[i + 1];
                    ssimG += this.calculateSSIMForChannel(g1, g2, C1, C2);
                    
                    const b1 = data1[i + 2];
                    const b2 = data2[i + 2];
                    ssimB += this.calculateSSIMForChannel(b1, b2, C1, C2);
                }
                
                const totalPixels = canvas1.width * canvas1.height;
                const msssim = (ssimR + ssimG + ssimB) / (3 * totalPixels);
                
                resolve(Math.max(0, Math.min(1, msssim)));
            } catch (error) {
                console.error('SSIM calculation error:', error);
                resolve(0);
            }
        });
    }

    calculateSSIMForChannel(x, y, C1, C2) {
        const mx = x;
        const my = y;
        const vx = Math.pow(x - mx, 2);
        const vy = Math.pow(y - my, 2);
        const vxy = (x - mx) * (y - my);
        
        return ((2 * mx * my + C1) * (2 * vxy + C2)) / 
               ((mx * mx + my * my + C1) * (vx + vy + C2));
    }

    deleteImage() {
        this.originalImage.src = '';
        this.compressedImage.src = '';
        this.originalSize.textContent = '-';
        this.compressedSize.textContent = '-';
        this.compressionRatio.textContent = '-';
        this.psnrValue.textContent = '-';
        this.msssimValue.textContent = '-';
        this.deleteButton.style.display = 'none';
        this.fileInput.value = '';
    }

    getFileSizeInBytes(sizeStr) {
        const size = parseFloat(sizeStr.split(' ')[0]);
        const unit = sizeStr.split(' ')[1];
        switch(unit) {
            case 'KB': return Math.round(size * 1024);
            case 'MB': return Math.round(size * 1024 * 1024);
            default: return Math.round(size);
        }
    }
}

let bpgEncoder = null;

async function initBPGEncoder() {
    if (!bpgEncoder) {
        await LibBPG.load();
        bpgEncoder = new LibBPG();
    }
    return bpgEncoder;
}

document.addEventListener('DOMContentLoaded', () => {
    new ImageCompressor();
}); 