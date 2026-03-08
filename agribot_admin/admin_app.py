# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, abort
from minio import Minio
from pymilvus import MilvusClient
import os
from dotenv import load_dotenv
from datetime import timezone
from functools import wraps
import pytz

# 加载环境变量
load_dotenv()



# 管理员账号
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

# Minio 配置
MINIO_BUCKET = os.getenv('MINIO_BUCKET')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_DATABASE = os.getenv("MILVUS_DATABASE")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
MILVUS_INDEX_NAME = os.getenv("MILVUS_INDEX_NAME")



app = Flask(__name__)

#对session加密
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')


@app.before_request
def before_request():
    """在每个请求之前执行域名白名单检查，实则只进行了日志记录"""
    app.logger.info(f"before_request: {request.url}")

# 初始化 Minio 客户端
minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# 初始化 Milvus API
milvus_client = MilvusClient(
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
    token=f"{MILVUS_USER}:{MILVUS_PASSWORD}",
    db_name=MILVUS_DATABASE
)


def convert_utc_to_local(utc_datetime, timezone_name='Asia/Shanghai'):
    """
    将UTC时间转换为本地时间
    
    Args:
        utc_datetime: UTC时间对象（datetime）
        timezone_name: 目标时区名称，默认为'Asia/Shanghai'（中国标准时间）
    
    Returns:
        str: 格式化后的本地时间字符串（YYYY-MM-DD HH:MM:SS）
    """
    if not utc_datetime:
        return ''
    
    try:
        # 确保UTC时间有时区信息
        if utc_datetime.tzinfo is None:
            utc_time = utc_datetime.replace(tzinfo=timezone.utc)
        else:
            utc_time = utc_datetime
        
        # 转换为目标时区
        local_tz = pytz.timezone(timezone_name)
        local_time = utc_time.astimezone(local_tz)
        
        # 格式化为字符串
        return local_time.strftime('%Y-%m-%d %H:%M:%S')
    
    except Exception as e:
        app.logger.error(f"时区转换失败: {str(e)}")
        # 如果转换失败，返回原始时间的字符串格式
        return utc_datetime.strftime('%Y-%m-%d %H:%M:%S') if utc_datetime else ''

#定义一个装饰器
def login_required(f):#哪个函数调用这个装饰器，f就是哪个函数的名字
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login')) #这是让浏览器去请求 @app.route('/login', methods=['GET', 'POST'])
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """首页，重定向到登录页面或文件浏览页面"""
    if 'username' in session:
        return redirect(url_for('file_browser'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['username'] = username
            flash('登录成功！', 'success')
            return redirect(url_for('file_browser'))
        else:
            flash('用户名或密码错误', 'error')
    #如果是get请求，就是没提交表达，然后接着显示login这个界面

    return render_template('login.html')


@app.route('/logout')
def logout():
    """退出登录"""
    session.pop('username', None)
    flash('已退出登录', 'info')
    return redirect(url_for('login'))


# 企业微信登录相关接口，暂时不使用
#app.route('/wxwork/callback')

#app.route('/wxwork/login')

#app.route('/wxwork/verify', methods=['GET', 'POST'])

#app.route('/api/wxwork/user')

# 文件解析测试 API，暂时不使用
#app.route('/api/file/<path:file_path>/parse-test', methods=['POST'])

# 文件浏览页面
@app.route('/files')
@login_required
def file_browser():
    """文件浏览页面"""
    path = request.args.get('path', '')
    return render_template('file_browser.html', current_path=path)

# 获取文件列表 API
@app.route('/api/files')
@login_required
def api_files():
    """获取文件列表 API
    让前端像浏览文件系统一个浏览minio中的文件和文件夹"""
    path = request.args.get('path', '')
    
    try:
        # 测试 MinIO 连接
        if not minio_client.bucket_exists(MINIO_BUCKET):
            return jsonify({'error': f'桶 "{MINIO_BUCKET}" 不存在，请检查配置'}), 500
        
        # 获取 Minio 中的文件列表
        # 构建搜索前缀
        search_prefix = path if path == '' else (path if path.endswith('/') else path + '/')
        '''
        search_prefix做的操作是对用户输入的路径进行规范化
        path 是用户输入的路径：比如/api/files?path=docs 
        此时path就是docs， 
        search_prefix根据上一行代码就是 search_prefix/
        
        为什么要规范用户输入的路径呢？
            因为我们接下来的操作是拿出用户输入路径下的所有文件或者文件夹
            如果用户输入的路径不标准，我们后续拿到的这个路径下的文件或者文件夹就不太正确
        '''
        
        objects = minio_client.list_objects(MINIO_BUCKET, prefix=search_prefix, recursive=False) #不递归子目录
        #传进来的 search_prefix就相当于告诉minio我只要 search_prefix这个路径下的文件或者文件夹，相当于指定musql中的表名
        
        files = []
        folders = set()
        
        for obj in objects:
            # 跳过当前目录的标记对象
            if obj.object_name == search_prefix.rstrip('/'):    #如果对象名就是目录本身 → 跳过。 rstrip 删除字符串 右侧末尾的指定字符
                continue
            
            # 计算相对路径
            if search_prefix: #只要用户输入的不是空目录，就会一直执行这个if
                if not obj.object_name.startswith(search_prefix): #这个if不会执行，因为前面已经规范了objects中的文件名是以search_prefix开头的
                    continue
                relative_path = obj.object_name[len(search_prefix):] #切掉前缀，得到相对路径 'docs/file1.txt' → 'file1.txt'
            else:   #用户输入的是""就不用切了，直接返回用户输入的path即可
                relative_path = obj.object_name
            
            # 跳过空路径
            if not relative_path:
                continue
            
            if '/' in relative_path:
                # 这是一个子目录中的对象，我们只需要目录名而不是要子目录中的文件
                folder_name = relative_path.split('/')[0] #得到子目录的名字 'subfolder/file3.txt' → 'subfolder'
                if folder_name and folder_name not in folders: #set去重 防止同一个目录重复添加
                    folders.add(folder_name)
                    folder_path = path + '/' + folder_name if path else folder_name #if path表示当前目录不是根目录，即当前目录非空
                    files.append({
                        'name': folder_name,
                        'path': folder_path,
                        'type': 'folder',
                        'size': 0,
                        'modified': ''
                    })
            else:
                # 这是当前目录下的文件
                files.append({
                    'name': relative_path,
                    'path': obj.object_name,
                    'type': 'file',
                    'size': obj.size,
                    'modified': convert_utc_to_local(obj.last_modified)
                })
        
        return jsonify({'files': files})
        '''
        {
            "files": [
                {"name": "file1.txt", "path": "docs/file1.txt", "type": "file", "size": 1024, "modified": "2026-01-07 18:00:00"},
                {"name": "file2.txt", "path": "docs/file2.txt", "type": "file", "size": 2048, "modified": "2026-01-07 19:00:00"},
                {"name": "subfolder", "path": "docs/subfolder", "type": "folder", "size": 0, "modified": ""}
            ]
        }'''
    except Exception as e:
        app.logger.error(f"获取文件列表失败: {str(e)}")
        if "SignatureDoesNotMatch" in str(e):
            return jsonify({'error': 'MinIO 认证失败，请检查 ACCESS_KEY 和 SECRET_KEY 配置'}), 500
        elif "Connection" in str(e):
            return jsonify({'error': f'无法连接到 MinIO 服务器 ({os.getenv("ENDPOINT")})，请检查网络和地址配置'}), 500
        else:
            return jsonify({'error': f'获取文件列表失败: {str(e)}'}), 500


# 文件详情页面
@app.route('/file/<path:file_path>')
@login_required
def file_detail(file_path):
    """文件详情页面"""
    return render_template('file_detail.html', file_path=file_path)


# 获取一个文件的详细信息，包括 MinIO 存储信息和 Milvus 索引信息。
@app.route('/api/file/<path:file_path>')
@login_required
def api_file_detail(file_path):
    """获取文件详情 API"""
    try:
        # URL 解码文件路径（处理可能的双重编码）
        from urllib.parse import unquote
        # 先尝试解码一次，如果结果仍然是编码格式，再解码一次
        decoded_path = unquote(file_path)
        if '%' in decoded_path:
            decoded_path = unquote(decoded_path)
        file_path = decoded_path    #/api/file/docs/file1.txt → file_path = 'docs/file1.txt'
        
        app.logger.debug(f"原始路径: {file_path}")
        app.logger.debug(f"解码后路径: {decoded_path}")

        # 从 Minio 获取文件基本信息
        try:
            file_stat = minio_client.stat_object(MINIO_BUCKET, file_path)
            
            file_info = {
                'doc_name': os.path.basename(file_path),
                'doc_path_name': file_path,
                'file_size': file_stat.size,
                'file_md5': file_stat.etag.strip('"'),
                'last_modified': convert_utc_to_local(file_stat.last_modified),
                'indexed': False,
                'index_time': '',
                'chunks': []
            }
        except Exception as e:
            app.logger.error(f"获取文件信息失败: {str(e)}")
            if "SignatureDoesNotMatch" in str(e):
                return jsonify({'error': 'MinIO 认证失败，请检查 ACCESS_KEY 和 SECRET_KEY 配置'}), 500
            elif "NoSuchKey" in str(e) or "not found" in str(e).lower():
                return jsonify({'error': f'文件不存在: {file_path}'}), 404
            else:
                return jsonify({'error': f'获取文件信息失败: {str(e)}'}), 500
        
        # 看看这个文件是否已经被向量化并存入 Milvus 索引
        try:
            milvus_client.load_collection(collection_name=MILVUS_COLLECTION)
            filter_expr = f'doc_path_name == "{file_path}"'
            
            results = milvus_client.query(
                collection_name=MILVUS_COLLECTION,
                filter=filter_expr,
                output_fields=["doc_name", "doc_path_name", "doc_type", "doc_md5", "doc_length", "content", "embedding_model"],
                limit=100
            )
            '''
                results = [
                    {
                        "doc_name": "file1.txt",
                        "doc_path_name": "docs/file1.txt",
                        "doc_type": "txt",
                        "doc_md5": "abc123",
                        "doc_length": 1024,
                        "content": "第一部分内容",
                        "embedding_model": "text-embedding-3-small"
                    },
                    {
                        "doc_name": "file1.txt",
                        "doc_path_name": "docs/file1.txt",
                        "doc_type": "txt",
                        "doc_md5": "abc123",
                        "doc_length": 512,
                        "content": "第二部分内容",
                        "embedding_model": "text-embedding-3-small"
                    }
                ]'''
            
            if results:
                file_info['indexed'] = True #如果文件已经被索引，标记为True
                # 取第一条记录的基本信息
                first_record = results[0] #一个pdf可能被分成多个chunk。我们为了获取这个文件的基础信息 取出第一个chunk就行
                file_info['doc_type'] = first_record.get('doc_type', '')
                file_info['indexed_md5'] = first_record.get('doc_md5', '')
                file_info['doc_length'] = first_record.get('doc_length', 0)
                file_info['embedding_model'] = first_record.get('embedding_model', '')
                
                # 处理所有分片（chunk）
                for result in results:
                    file_info['chunks'].append({
                        'content': result.get('content', ''),
                        'length': len(result.get('content', ''))
                    })
        except Exception as e:
            print(f"查询 Milvus 失败: {e}")
        
        return jsonify(file_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 文件下载 API
@app.route('/api/file/<path:file_path>/download')
@login_required
def api_download_file(file_path):
    """文件下载 API"""
    try:
        # URL 解码文件路径（处理可能的双重编码）
        from urllib.parse import unquote
        decoded_path = unquote(file_path)
        if '%' in decoded_path:
            decoded_path = unquote(decoded_path)
        file_path = decoded_path
        
        app.logger.info(f"下载文件: {file_path}")
        
        # 检查文件是否存在
        try:
            file_stat = minio_client.stat_object(MINIO_BUCKET, file_path)
        except Exception as e:
            if "NoSuchKey" in str(e) or "not found" in str(e).lower():
                return jsonify({'error': f'文件不存在: {file_path}'}), 404
            else:
                return jsonify({'error': f'获取文件信息失败: {str(e)}'}), 500
        
        # 从 MinIO 获取文件
        from flask import Response
        import io
        
        try:
            response = minio_client.get_object(MINIO_BUCKET, file_path)
            file_data = response.data
            
            # 获取文件名
            filename = os.path.basename(file_path)
            
            # 设置正确的 Content-Type
            content_type = file_stat.content_type or 'application/octet-stream'
            
            # 对文件名进行URL编码以支持中文字符
            from urllib.parse import quote
            encoded_filename = quote(filename, safe='')
            
            # 创建响应
            return Response(
                file_data,
                mimetype=content_type,
                headers={
                    'Content-Disposition': f'attachment; filename*=UTF-8\'\'{encoded_filename}',
                    'Content-Length': str(len(file_data)),
                    'Cache-Control': 'no-cache'
                }
            )
            
        except Exception as e:
            app.logger.error(f"下载文件失败: {str(e)}")
            return jsonify({'error': f'下载文件失败: {str(e)}'}), 500
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()
        
    except Exception as e:
        app.logger.error(f"下载请求处理失败: {str(e)}")
        return jsonify({'error': f'下载请求处理失败: {str(e)}'}), 500

# 文件删除 API
@app.route('/api/file/<path:file_path>/delete', methods=['DELETE'])
@login_required
def api_delete_file(file_path):
    """文件删除 API"""
    try:
        # URL 解码文件路径（处理可能的双重编码）
        from urllib.parse import unquote
        decoded_path = unquote(file_path)
        if '%' in decoded_path:
            decoded_path = unquote(decoded_path)
        file_path = decoded_path
        
        app.logger.info(f"删除文件: {file_path}")
        
        # 检查文件是否存在
        try:
            file_stat = minio_client.stat_object(MINIO_BUCKET, file_path)
            file_name = os.path.basename(file_path)
        except Exception as e:
            if "NoSuchKey" in str(e) or "not found" in str(e).lower():
                return jsonify({'error': f'文件不存在: {file_path}'}), 404
            else:
                return jsonify({'error': f'获取文件信息失败: {str(e)}'}), 500
        
        # 从 MinIO 删除文件
        try:
            minio_client.remove_object(MINIO_BUCKET, file_path)
            app.logger.info(f"文件删除成功: {file_path}")
            
            # 如果文件在 Milvus 中有索引，也删除索引记录
            try:
                milvus_client.load_collection(collection_name=MILVUS_COLLECTION)
                filter_expr = f'doc_path_name == "{file_path}"'
                
                # 查询是否存在索引记录
                results = milvus_client.query(
                    collection_name=MILVUS_COLLECTION,
                    filter=filter_expr,
                    output_fields=["id"],
                    limit=1000
                )
                
                if results:
                    # 删除 Milvus 中的记录
                    ids_to_delete = [str(result['id']) for result in results]
                    milvus_client.delete(
                        collection_name=MILVUS_COLLECTION,
                        filter=f'id in {ids_to_delete}'
                    )
                    app.logger.info(f"已删除 Milvus 中的 {len(ids_to_delete)} 条记录")
                
            except Exception as e:
                app.logger.warning(f"删除 Milvus 索引时出错 (文件已删除): {str(e)}")
            
            return jsonify({
                'success': True,
                'message': f'文件 "{file_name}" 删除成功',
                'file_name': file_name,
                'file_path': file_path
            })
            
        except Exception as e:
            app.logger.error(f"删除文件失败: {str(e)}")
            return jsonify({'error': f'删除文件失败: {str(e)}'}), 500
        
    except Exception as e:
        app.logger.error(f"删除请求处理失败: {str(e)}")
        return jsonify({'error': f'删除请求处理失败: {str(e)}'}), 500

# 创建目录 API
@app.route('/api/create-directory', methods=['POST'])
@login_required
def api_create_directory():
    """创建目录 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
        
        current_path = data.get('path', '')
        dir_name = data.get('name', '').strip()
        
        # 验证目录名称
        if not dir_name:
            return jsonify({'success': False, 'error': '目录名称不能为空'}), 400
        
        # 检查非法字符
        import re
        if re.search(r'[/\\:*?"<>|]', dir_name):
            return jsonify({'success': False, 'error': '目录名称不能包含以下字符: / \\ : * ? " < > |'}), 400
        
        # 构建完整路径
        if current_path:
            full_path = current_path.rstrip('/') + '/' + dir_name + '/'
        else:
            full_path = dir_name + '/'
        
        app.logger.info(f"创建目录: {full_path}")
        
        # 检查目录是否已存在
        try:
            minio_client.stat_object(MINIO_BUCKET, full_path)
            return jsonify({'success': False, 'error': '目录已存在'}), 400
        except Exception:
            # 目录不存在，可以创建
            pass
        
        # 创建目录（通过上传一个空对象）
        from io import BytesIO
        minio_client.put_object(
            MINIO_BUCKET,
            full_path,
            BytesIO(b''),
            0,
            content_type='application/x-directory'
        )
        
        app.logger.info(f"目录创建成功: {full_path}")
        return jsonify({'success': True, 'message': '目录创建成功'})
        
    except Exception as e:
        app.logger.error(f"创建目录失败: {str(e)}")
        return jsonify({'success': False, 'error': f'创建目录失败: {str(e)}'}), 500

# 删除目录 API
@app.route('/api/delete-directory', methods=['DELETE'])
@login_required
def api_delete_directory():
    """删除目录 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
        
        dir_path = data.get('path', '').strip()
        
        # 验证路径
        if not dir_path:
            return jsonify({'success': False, 'error': '目录路径不能为空'}), 400
        
        # 确保路径以 / 结尾
        if not dir_path.endswith('/'):
            dir_path += '/'
        
        app.logger.info(f"删除目录: {dir_path}")
        
        # 检查目录是否存在
        try:
            minio_client.stat_object(MINIO_BUCKET, dir_path)
        except Exception:
            return jsonify({'success': False, 'error': '目录不存在'}), 404
        
        # 检查目录是否为空（不包含任何文件或子目录）
        objects = list(minio_client.list_objects(MINIO_BUCKET, prefix=dir_path, recursive=True))
        
        # 过滤掉目录本身
        content_objects = [obj for obj in objects if obj.object_name != dir_path]
        
        if content_objects:
            return jsonify({'success': False, 'error': '目录不为空，无法删除'}), 400
        
        # 删除目录
        minio_client.remove_object(MINIO_BUCKET, dir_path)
        
        app.logger.info(f"目录删除成功: {dir_path}")
        return jsonify({'success': True, 'message': '目录删除成功'})
        
    except Exception as e:
        app.logger.error(f"删除目录失败: {str(e)}")
        return jsonify({'success': False, 'error': f'删除目录失败: {str(e)}'}), 500

# 上传文件 API
@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload_files():
    """文件上传 API"""
    try:
        # 获取上传路径
        upload_path = request.form.get('path', '')
        
        # 检查是否有文件
        if 'files' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': '没有选择有效的文件'}), 400
        
        app.logger.info(f"开始上传 {len(files)} 个文件到路径: {upload_path}")
        
        uploaded_files = []
        failed_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            try:
                # 构建完整的对象路径
                if upload_path:
                    object_name = f"{upload_path}/{file.filename}"
                else:
                    object_name = file.filename
                
                # 检查文件是否已存在
                try:
                    minio_client.stat_object(MINIO_BUCKET, object_name)
                    failed_files.append({
                        'filename': file.filename,
                        'error': '文件已存在'
                    })
                    continue
                except:
                    # 文件不存在，可以上传
                    pass
                
                # 上传文件到 MinIO
                file.seek(0)  # 重置文件指针
                file_data = file.read()
                file_size = len(file_data)
                
                # 检查文件大小限制 (500MB)
                max_size = 500 * 1024 * 1024
                if file_size > max_size:
                    failed_files.append({
                        'filename': file.filename,
                        'error': f'文件过大 ({file_size / 1024 / 1024:.2f} MB > 500 MB)'
                    })
                    continue
                
                # 重置文件指针并上传
                file.seek(0)
                minio_client.put_object(
                    MINIO_BUCKET,
                    object_name,
                    file,
                    file_size,
                    content_type=file.content_type or 'application/octet-stream'
                )
                
                uploaded_files.append({
                    'filename': file.filename,
                    'object_name': object_name,
                    'size': file_size
                })
                
                app.logger.info(f"文件上传成功: {object_name}")
                
            except Exception as e:
                app.logger.error(f"上传文件 {file.filename} 失败: {str(e)}")
                failed_files.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        # 构造返回结果
        result = {
            'success': len(uploaded_files) > 0,
            'uploaded_count': len(uploaded_files),
            'failed_count': len(failed_files),
            'uploaded_files': uploaded_files,
            'failed_files': failed_files,
            'message': f'成功上传 {len(uploaded_files)} 个文件'
        }
        
        if failed_files:
            result['message'] += f'，{len(failed_files)} 个文件上传失败'
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"文件上传处理失败: {str(e)}")
        return jsonify({'error': f'文件上传处理失败: {str(e)}'}), 500



if __name__ == '__main__':
    
    # 获取主机和端口配置
    host = os.getenv('FLASK_HOST')
    port = int(os.getenv('FLASK_PORT'))


    print("=" * 60)
    print("Agribot 管理后台启动中...")
    print(f"访问端口: {port}")
    print("=" * 60)
    
    app.run(
        host=host,
        port=port,
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        threaded=True,
        processes=1
    ) 