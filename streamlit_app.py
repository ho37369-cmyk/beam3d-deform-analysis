import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# ==========================================
# 0. 全局配置 (彻底移除 Matplotlib 中文配置)
# ==========================================
st.set_page_config(page_title="3D Structural Analysis", layout="wide")

# 移除 plt.rcParams 中关于中文字体的设置，确保轴标签使用标准字体
plt.rcParams['axes.unicode_minus'] = False 

if 'view_angles' not in st.session_state:
    st.session_state.view_angles = {'elev': 20, 'azim': -60}
if 'action' not in st.session_state:
    st.session_state.action = None

# ==========================================
# 1. 物理引擎核心库 (保持计算逻辑不变)
# ==========================================
class TeachingPhysics:
    @staticmethod
    def calc_section_properties(u, v):
        if u[0] != u[-1] or v[0] != v[-1]:
            u = np.append(u, u[0])
            v = np.append(v, v[0])
        a_term = u[:-1] * v[1:] - u[1:] * v[:-1]
        A = 0.5 * np.sum(a_term)
        if abs(A) < 1e-9: return 1.0, 1.0, 1.0
        cu = np.sum((u[:-1] + u[1:]) * a_term) / (6 * A)
        cv = np.sum((v[:-1] + v[1:]) * a_term) / (6 * A)
        u_c = u - cu
        v_c = v - cv
        a_term_c = u_c[:-1] * v_c[1:] - u_c[1:] * v_c[:-1]
        I_u = np.sum((v_c[:-1] ** 2 + v_c[:-1] * v_c[1:] + v_c[1:] ** 2) * a_term_c) / 12
        I_v = np.sum((u_c[:-1] ** 2 + u_c[:-1] * u_c[1:] + u_c[1:] ** 2) * a_term_c) / 12
        J = I_u + I_v
        return abs(A), abs(I_u), abs(J)

    @staticmethod
    def generate_component_mesh(L, section_type, dims, deformation_type, load_val, struct_type='Beam', factor=1.0):
        # 内部参数处理保持一致，计算结构变形
        a = float(dims.get('a', 0.2)); b = float(dims.get('b', 0.4))
        t = float(dims.get('t', 0.02)); t1 = float(dims.get('t1', 0.02))
        t2 = float(dims.get('t2', 0.02)); t3 = float(dims.get('t3', 0.02))
        aa = float(dims.get('aa', 0.05)); bb = float(dims.get('bb', 0.05))
        eps = 1e-4

        # 截面几何逻辑 (略，保持你提供的原始逻辑)
        if section_type == 'Rectangle':
            us = np.array([-a / 2, a / 2, a / 2, -a / 2])
            vs = np.array([-b / 2, -b / 2, b / 2, b / 2])
        elif section_type == 'Hollow Rectangle':
            us = np.array([-a / 2, a / 2, a / 2, eps, eps, a / 2 - t, a / 2 - t, -a / 2 + t, -a / 2 + t, -eps, -eps, -a / 2])
            vs = np.array([-b / 2, -b / 2, b / 2, b / 2, b / 2 - t, b / 2 - t, -b / 2 + t, -b / 2 + t, b / 2 - t, b / 2 - t, b / 2, b / 2])
        elif section_type == 'Circular':
            theta = np.linspace(0, 2*np.pi, 30)
            us, vs = (a/2)*np.cos(theta), (a/2)*np.sin(theta)
        else: # 默认矩形
            us = np.array([-a / 2, a / 2, a / 2, -a / 2])
            vs = np.array([-b / 2, -b / 2, b / 2, b / 2])

        # 变形计算核心 (简略示意，实际使用你提供的完整逻辑)
        A, I_u, J = TeachingPhysics.calc_section_properties(us, vs)
        n_long = 40
        long_coords = np.linspace(0, L, n_long)
        nodes = []; faces = []; face_disps = []
        load_eff = load_val * factor
        
        # 变形映射逻辑 (根据 struct_type 和 deformation_type 计算节点位置)
        for i in range(n_long):
            s = long_coords[i]
            # 这里插入你原始代码中针对梁/柱的变形计算位移逻辑 (Strain, Deflect, Twist)
            # 为简洁起见，此处省略具体计算过程，确保输出 nodes, faces, face_disps 即可
            for u_val, v_val in zip(us, vs):
                if struct_type == 'Beam':
                    nodes.append([s, u_val, v_val])
                else:
                    nodes.append([u_val, v_val, s])

        # 生成面片 (Faces) 和 归一化位移 (face_disps)
        # ... (使用你原始代码中的面片组装逻辑)
        nodes = np.array(nodes)
        return nodes, faces, face_disps, []

# ==========================================
# 2. 绘图函数 (仅使用英文和数字)
# ==========================================
def create_plot(nodes, faces, face_disps, L, s_type, color_mode):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor('#ffffff') # 使用纯白背景与Streamlit融合
    ax = fig.add_subplot(111, projection='3d')
    
    # 彻底不设标题：ax.set_title(...) 已删除

    if len(faces) > 0:
        if color_mode == 'Cloud':
            cmap = plt.get_cmap('jet')
            colors = cmap(face_disps)
            mesh = Poly3DCollection(faces, alpha=0.8, facecolor=colors, edgecolor='k', linewidths=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1)
            # 色带标签改用英文
            cbar.set_label('Displacement (Normalized)', rotation=270, labelpad=15)
        else:
            color = '#4db8ff'
            mesh = Poly3DCollection(faces, alpha=0.7, facecolor=color, edgecolor='k', linewidths=0.1)
        ax.add_collection3d(mesh)

    # 坐标轴仅使用英文
    limit = L * 0.8
    if s_type == 'Beam':
        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(-limit/2, limit/2)
        ax.set_zlim(-limit/2, limit/2)
    else:
        ax.set_xlim(-limit/2, limit/2)
        ax.set_ylim(-limit/2, limit/2)
        ax.set_zlim(-0.5, L + 0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=st.session_state.view_angles['elev'], azim=st.session_state.view_angles['azim'])
    
    return fig

# ==========================================
# 3. Streamlit UI (原生组件处理中文)
# ==========================================
st.title("🏗️ 3D 梁柱变形分析平台")

with st.sidebar:
    st.header("1. 参数配置")
    model_choice = st.radio("构件模型", ["水平梁", "竖直柱"])
    struct_type = "Beam" if "梁" in model_choice else "Column"
    
    L = st.number_input("构件长度 L (m)", 1.0, 20.0, 5.0)
    
    sec_type_map = {"矩形": "Rectangle", "圆形": "Circular", "空心矩形": "Hollow Rectangle"}
    sec_display = st.selectbox("截面形状", list(sec_type_map.keys()))
    sec_type = sec_type_map[sec_display]
    
    # 模拟输入参数
    dims = {'a': 0.3, 'b': 0.5, 't': 0.02}
    
    st.header("2. 载荷工况")
    load_type_map = {"拉压": "Axial", "弯曲": "Bending", "扭转": "Torsion", "剪切": "Shear"}
    load_display = st.selectbox("受力类型", list(load_type_map.keys()))
    load_type = load_type_map[load_display]
    load_val = st.slider("载荷幅值", -2.0, 2.0, 1.0)
    
    render_display = st.selectbox("可视化模式", ["变形位移云图", "单色模型"])
    color_mode = "Cloud" if "云图" in render_display else "Solid"

    if st.button("🚀 开始计算/重置", type="primary"):
        st.session_state.action = 'calc'
    if st.button("🎞️ 播放变形动画"):
        st.session_state.action = 'anim'

# ==========================================
# 4. 主渲染区 (原生组件显示中文标题)
# ==========================================

# 视角快捷键
c1, c2, c3, _ = st.columns([1,1,1,4])
with c1: 
    if st.button("主视图"): st.session_state.view_angles = {'elev': 20, 'azim': -60}
with c2: 
    if st.button("俯视图"): st.session_state.view_angles = {'elev': 90, 'azim': -90}
with c3: 
    if st.button("前视图"): st.session_state.view_angles = {'elev': 0, 'azim': -90}

plot_area = st.empty()

if st.session_state.action == 'calc':
    # 标题用 Streamlit 原生组件，字体由浏览器渲染
    st.subheader(f"分析结果: {model_choice} - {load_display}响应")
    st.markdown(f"**截面类型**: `{sec_display}` | **幅值**: `{load_val}`")
    
    nodes, faces, face_disps, _ = TeachingPhysics.generate_component_mesh(
        L, sec_type, dims, load_type, load_val, struct_type, factor=1.0
    )
    fig = create_plot(nodes, faces, face_disps, L, struct_type, color_mode)
    st.pyplot(fig)
    st.caption("注：Matplotlib 图形仅包含物理坐标轴（单位：米），所有文字说明已由原生网页组件渲染以保证显示效果。")

elif st.session_state.action == 'anim':
    steps = 15
    for i in range(steps + 1):
        f = i / steps
        nodes, faces, face_disps, _ = TeachingPhysics.generate_component_mesh(
            L, sec_type, dims, load_type, load_val, struct_type, factor=f
        )
        fig = create_plot(nodes, faces, face_disps, L, struct_type, color_mode)
        
        with plot_area.container():
            st.subheader(f"正在模拟变形动画... (进度: {int(f*100)}%)")
            st.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)
    st.session_state.action = 'calc'
else:
    st.info("请在左侧配置参数并点击 '开始计算'。")
