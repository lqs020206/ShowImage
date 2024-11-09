import http.server
import socketserver
import subprocess
import os
import tempfile
import base64
import numpy as np
from PIL import Image
import io
from urllib.parse import parse_qs
import json
import math
import logging
import traceback
import sys
import os.path
import torch
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from net.network import WITT  # 需要确保net目录下有network.py
import torch.nn as nn
from loss.distortion import MS_SSIM
import argparse
import torchvision.transforms as transforms
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 修改BPG工具的路径定义
# 优先使用环境变量中的路径，如果找不到再使用本地路径
def find_bpg_tools():
    try:
        bpgenc_path = subprocess.check_output(['where', 'bpgenc']).decode().strip().split('\n')[0]
        bpgdec_path = subprocess.check_output(['where', 'bpgdec']).decode().strip().split('\n')[0]
        logger.debug(f"Found bpgenc in PATH: {bpgenc_path}")
        logger.debug(f"Found bpgdec in PATH: {bpgdec_path}")
        return bpgenc_path, bpgdec_path
    except subprocess.CalledProcessError:
        # 如果在环境变量中找不到，使用本地路径
        local_bpgenc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libbpg', 'bpgenc.exe')
        local_bpgdec = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libbpg', 'bpgdec.exe')
        logger.debug(f"Using local bpg tools: {local_bpgenc}, {local_bpgdec}")
        return local_bpgenc, local_bpgdec

# 获取BPG工具路径
BPGENC_PATH, BPGDEC_PATH = find_bpg_tools()

class ImageQuality:
    @staticmethod
    def calculate_psnr(img1, img2):
        """使用skimage计算PSNR"""
        try:
            return psnr(img1, img2)
        except Exception as e:
            logger.error(f"PSNR calculation error: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_msssim(img1, img2):
        """使用skimage计算MSSSIM"""
        try:
            # 设置一个合适的win_size，避免超过图像尺寸
            win_size = 3  # 使用较小的窗口大小
            return ssim(img1, img2, multichannel=True, win_size=win_size)
        except Exception as e:
            logger.error(f"MSSSIM calculation error: {str(e)}")
            return 0.0

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        """计算MS-SSIM（Multi-Scale Structural Similarity）"""
        def gaussian_kernel(size=11, sigma=1.5):
            """创建高斯核"""
            x = np.linspace(-size//2, size//2, size)
            g = np.exp(-x**2 / (2*sigma**2))
            return np.outer(g, g)

        def ssim_at_scale(img1, img2, kernel):
            """计算特定尺度的SSIM"""
            # 常数
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            # 使用高斯核进行平滑
            mu1 = np.apply_over_axes(np.mean, img1, [-2, -1])
            mu2 = np.apply_over_axes(np.mean, img2, [-2, -1])
            
            sigma1_sq = np.apply_over_axes(np.var, img1, [-2, -1])
            sigma2_sq = np.apply_over_axes(np.var, img2, [-2, -1])
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(-2, -1))
            
            # 计算SSIM
            num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
            
            return np.mean(num / den)

        def downsample(img):
            """下采样图像"""
            return img[::2, ::2]

        # 转换为float类型
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # 创建高斯核
        kernel = gaussian_kernel()
        
        # 权重，从细到粗的尺度
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        scales = len(weights)
        
        mssim = []
        # 对每个颜色通道分别计算
        for channel in range(3):  # RGB
            im1 = img1[:,:,channel]
            im2 = img2[:,:,channel]
            
            # 计算多尺度SSIM
            for scale in range(scales):
                ssim = ssim_at_scale(im1, im2, kernel)
                mssim.append(ssim * weights[scale])
                
                if scale < scales - 1:
                    im1 = downsample(im1)
                    im2 = downsample(im2)
        
        # 返回加权平均的MS-SSIM
        return np.sum(mssim) / 3  # 除以3是因为有3个颜色通道

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        try:
            # 获取Content-Type和boundary
            content_type = self.headers.get('Content-Type', '')
            logger.debug(f"Content-Type: {content_type}")
            
            # 读取请求内容
            content_length = int(self.headers.get('Content-Length', 0))
            logger.debug(f"Content-Length: {content_length}")
            
            if content_length == 0:
                raise ValueError("No content received")
                
            post_data = self.rfile.read(content_length)
            logger.debug(f"Received {len(post_data)} bytes of data")
            
            # 解析参数
            query_string = self.path.split('?')[1] if '?' in self.path else ''
            params = parse_qs(query_string)
            quality = params.get('quality', ['28'])[0]
            format = params.get('format', ['bpg'])[0]
            snr = float(params.get('snr', ['10'])[0])
            channel = int(params.get('channel', ['96'])[0])
            
            # 处理multipart/form-data
            if content_type.startswith('multipart/form-data'):
                import cgi
                import io
                
                environ = {
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                    'CONTENT_LENGTH': str(content_length)
                }
                
                form = cgi.FieldStorage(
                    fp=io.BytesIO(post_data),
                    headers=self.headers,
                    environ=environ
                )
                
                if 'image' not in form:
                    raise ValueError("No image file received in form data")
                
                # 获取图片数据
                image_item = form['image']
                post_data = image_item.file.read()
                logger.debug(f"Extracted image data: {len(post_data)} bytes")
            
            logger.debug(f"Processing request: quality={quality}, format={format}")
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, 'input.png')
            output_path = os.path.join(temp_dir, f'output.{format}')
            decoded_path = os.path.join(temp_dir, 'decoded.png')
            
            try:
                # 保存原始图像
                with open(input_path, 'wb') as f:
                    f.write(post_data)
                logger.debug(f"Saved input image to {input_path}")
                
                if format == 'witt':
                    # 设置随机种子和设备
                    torch.manual_seed(1024)  # 使用固定种子确保结果可重现
                    device = torch.device('cpu')
                    
                    # 创建配置
                    witt_config = WITTConfig(channel=int(channel))
                    witt_config.logger = logger
                    witt_config.device = device
                    witt_config.CUDA = False  # 确保使用CPU
                    
                    # 创建参数命名空间
                    witt_args = argparse.Namespace(
                        trainset='DIV2K',
                        testset='DIV2K',
                        distortion_metric='MSE',
                        model='WITT',
                        channel_type='awgn',
                        C=int(channel),
                        multiple_snr=str(int(float(snr)))
                    )
                    
                    # 创建模型实例
                    witt_model = WITT(witt_args, witt_config)
                    
                    # 加载预训练权重
                    model_path = f"./awgn/DIV2K/WITT_AWGN_DIV2K_fixed_snr{int(float(snr))}_psnr_C{channel}.model"
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"找不到对应的预训练模型文件: {model_path}")
                    
                    # 加载权重到CPU
                    pretrained = torch.load(model_path, map_location=device)
                    witt_model.load_state_dict(pretrained, strict=True)
                    del pretrained  # 释放内存
                    
                    # 设���为评估模式
                    witt_model.eval()
                    
                    # 读取和预处理图像
                    original_img = Image.open(input_path)
                    if original_img.mode != 'RGB':
                        original_img = original_img.convert('RGB')
                    
                    # 裁剪为256x256
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor()
                    ])
                    
                    # 保存裁剪后的原图，用于后续计算PSNR和MSSSIM
                    cropped_original = transforms.ToPILImage()(transform(original_img))
                    cropped_path = os.path.join(temp_dir, 'cropped.png')
                    cropped_original.save(cropped_path)
                    
                    # 转换为tensor进行压缩
                    input_tensor = transform(original_img).unsqueeze(0)
                    
                    # WITT压缩
                    with torch.no_grad():
                        # 使用forward方法进行压缩
                        recon_image, CBR, SNR, mse, loss_G = witt_model(input_tensor, int(float(snr)))
                    
                    # 保存重建图像
                    recon_image = recon_image.squeeze(0)
                    recon_image = transforms.ToPILImage()(recon_image.clamp(0., 1.))
                    recon_image.save(decoded_path)
                    
                    # 使用裁剪后的原图和重建图像计算质量指标
                    original_img = np.array(Image.open(cropped_path))  # 使用裁剪后的原图
                    decoded_img = np.array(Image.open(decoded_path))
                    
                    # 使用与BPG相同的方式计算质量指标
                    psnr_value = ImageQuality.calculate_psnr(original_img, decoded_img)
                    msssim_value = ImageQuality.calculate_msssim(original_img, decoded_img)
                    
                    # 计算压缩率 = (channel*256)/(256*256*3*8)
                    channel_size = channel * 256  # channel * 256 bits
                    total_bits = 256 * 256 * 3 * 8  # 图像总比特数
                    compression_ratio = round((channel_size / total_bits) * 100, 3)  # 转换为百分比
                    
                    # 使用裁剪后的原图大小
                    original_size = os.path.getsize(cropped_path)
                    compressed_size = int(original_size * compression_ratio / 100)
                    
                    # 转换为base64
                    with open(decoded_path, 'rb') as f:
                        decoded_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    # 返回结果
                    result = {
                        'imageData': decoded_data,
                        'compressedSize': compressed_size,
                        'originalSize': original_size,
                        'compressionRatio': compression_ratio,
                        'psnr': psnr_value,
                        'msssim': msssim_value,
                        'debug': {
                            'dimensions': {
                                'width': original_img.shape[1],
                                'height': original_img.shape[0]
                            },
                            'snr': int(float(snr)),
                            'channel': int(channel)
                        }
                    }
                    
                    # 发送响应
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                    logger.debug("WITT compression completed successfully")
                
                elif format == 'bpg':
                    # 执行BPG压缩
                    cmd = [BPGENC_PATH, '-q', quality, '-o', output_path, input_path]
                    logger.debug(f"Running command: {' '.join(cmd)}")
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                        if result.returncode != 0:
                            raise RuntimeError(f"bpgenc failed: {result.stderr}")
                        logger.debug("BPG compression completed")
                        
                        # 解码BPG
                        cmd = [BPGDEC_PATH, '-o', decoded_path, output_path]
                        logger.debug(f"Running command: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                        if result.returncode != 0:
                            raise RuntimeError(f"bpgdec failed: {result.stderr}")
                        logger.debug("BPG decoding completed")
                        
                        # 计算质量指
                        original_img = np.array(Image.open(input_path))
                        decoded_img = np.array(Image.open(decoded_path))
                        
                        psnr = ImageQuality.calculate_psnr(original_img, decoded_img)
                        msssim = ImageQuality.calculate_msssim(original_img, decoded_img)
                        
                        # 获取文件大小和算压缩率
                        original_size = os.path.getsize(input_path)  # 单位：字节
                        compressed_size = os.path.getsize(output_path)  # 单位：字节

                        # 压缩率 = 压缩后大小/压缩前大小 * 100%
                        compression_ratio = round((compressed_size / original_size) * 100, 3)

                        # 让我们印详细信息来调试
                        logger.debug(f"Original size: {original_size} bytes")
                        logger.debug(f"Compressed size: {compressed_size} bytes")
                        logger.debug(f"Compression ratio: {compression_ratio}%")

                        # 验证计算
                        # 1.47 KB = 1.47 * 1024 = 1505.28 bytes
                        # 839.98 KB = 839.98 * 1024 = 860333.12 bytes
                        # 压缩率应该是 (1505.28 / 860333.12) * 100 = 0.175%
                        
                        # 将解码后的图像转为base64
                        with open(decoded_path, 'rb') as f:
                            decoded_data = base64.b64encode(f.read()).decode('utf-8')
                        
                        # 返回结果
                        result = {
                            'imageData': decoded_data,
                            'compressedSize': compressed_size,
                            'originalSize': original_size,
                            'compressionRatio': compression_ratio,  # 这是百分比形式
                            'psnr': psnr,
                            'msssim': msssim,
                            'debug': {
                                'dimensions': {
                                    'width': original_img.shape[1],
                                    'height': original_img.shape[0]
                                },
                                'quality': quality
                            }
                        }
                        
                        # 发送响应
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode())
                        logger.debug("Response sent successfully")
                        
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Command execution failed: {e.stderr}")
                        raise RuntimeError(f"Command execution failed: {e.stderr}")
                    except Exception as e:
                        logger.error(f"Error during BPG processing: {str(e)}")
                        raise
                elif format == 'jpeg':
                    # 使用 PIL 进行 JPEG 压缩
                    original_img = Image.open(input_path)
                    
                    # 如果图像是RGBA模式，转换为RGB
                    if original_img.mode == 'RGBA':
                        # 创建白色背景
                        background = Image.new('RGB', original_img.size, (255, 255, 255))
                        # 将原图复制到白色背景上
                        background.paste(original_img, mask=original_img.split()[3])  # 使用alpha通道作为mask
                        original_img = background
                    elif original_img.mode != 'RGB':
                        # 其他模式也转换为RGB
                        original_img = original_img.convert('RGB')
                    
                    # 保存为 JPEG，quality 范围是 1-100
                    original_img.save(output_path, 'JPEG', quality=int(quality))
                    logger.debug("JPEG compression completed")
                    
                    # 读取压缩后的图像
                    decoded_img = np.array(Image.open(output_path))
                    original_img_array = np.array(original_img)  # 使用转换后的图像
                    
                    # 计算质量指标
                    psnr = ImageQuality.calculate_psnr(original_img_array, decoded_img)
                    msssim = ImageQuality.calculate_msssim(original_img_array, decoded_img)
                    
                    # 获取文件大小和计算压缩率
                    original_size = os.path.getsize(input_path)
                    compressed_size = os.path.getsize(output_path)
                    compression_ratio = round((compressed_size / original_size) * 100, 3)
                    
                    logger.debug(f"Original size: {original_size} bytes")
                    logger.debug(f"Compressed size: {compressed_size} bytes")
                    logger.debug(f"Compression ratio: {compression_ratio}%")
                    
                    # 将压缩后的图像转为 base64
                    with open(output_path, 'rb') as f:
                        decoded_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    # 返回结果
                    result = {
                        'imageData': decoded_data,
                        'compressedSize': compressed_size,
                        'originalSize': original_size,
                        'compressionRatio': compression_ratio,
                        'psnr': psnr,
                        'msssim': msssim,
                        'debug': {
                            'dimensions': {
                                'width': original_img.size[0],  # 使用PIL Image的size属性
                                'height': original_img.size[1]
                            },
                            'quality': quality
                        }
                    }
                    
                    # 发送响应
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                    logger.debug("Response sent successfully")
                
            finally:
                # 清理临时文件
                for path in [input_path, output_path, decoded_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            logger.debug(f"Removed temporary file: {path}")
                        except Exception as e:
                            logger.error(f"Failed to remove {path}: {e}")
                try:
                    os.rmdir(temp_dir)
                    logger.debug(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove temp dir: {e}")
                
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_message = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
            self.wfile.write(error_message.encode())

# 添加WITT配置
class WITTConfig:
    def __init__(self, channel=16):
        self.seed = 1024
        self.pass_channel = True
        self.CUDA = False
        self.device = torch.device('cpu')
        self.image_dims = (3, 256, 256)
        self.C = channel
        self.logger = None
        self.trainset = 'DIV2K'
        self.testset = 'DIV2K'
        
        # 添加新的配置参数
        self.normalize = False
        self.downsample = 4  # DIV2K数据集使用的下采样率
        self.batch_size = 16
        self.norm = False
        
        # 更网络配置
        self.encoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]), 
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320], 
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            C=self.C,
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True
        )
        
        self.decoder_kwargs = dict(
            img_size=(self.image_dims[1], self.image_dims[2]),
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4], 
            C=self.C,
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True
        )

        # 添加训练相关配置
        self.learning_rate = 0.0001
        self.tot_epoch = 10000000
        self.min_lr = 1e-6
        self.grad_clip = 1.0
        self.patience = 5
        self.lr_factor = 0.5
        self.early_stop_patience = 15
        self.save_model_freq = 100

        # 添加logger相关配置
        self.print_step = 100
        self.plot_step = 10000
        self.filename = datetime.now().__str__()[:-7]
        self.workdir = './history/{}'.format(self.filename)
        self.log = self.workdir + '/Log_{}.log'.format(self.filename)
        self.samples = self.workdir + '/samples'
        self.models = self.workdir + '/models'

def load_weights(model, model_path):
    pretrained = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained, strict=True)
    del pretrained

# 移除全局的模型初始化，只保留服务器启动代码
PORT = 8080
Handler = CustomHTTPRequestHandler

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever() 