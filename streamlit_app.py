import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# ==========================================
# 0. 全局配置与状态初始化
# ==========================================
st.set_page_config(page_title="3D梁柱变形分析平台", layout="wide")

# 移除中文字体配置，确保 Linux 服务器渲染纯英文时不报错
plt.rcParams['axes.unicode_minus'] = False

if 'view_angles' not in st.session_state:
    st.session_state.view_angles = {'elev': 20, 'azim': -60}
if 'action' not in st.session_state:
    st.session_state.action = None

# ==========================================
# 1. 中英文字典映射 (用于图表内文字显示)
# ==========================================
EN_STRUCT = {"梁": "Beam", "柱": "Column"}
EN_LOAD = {"拉压": "Axial", "剪切": "Shear", "弯曲": "Bending", "扭转": "Torsion", "无": "None"}
EN_SEC = {
    "矩形": "Rectangular", "圆形": "Circular", "空心矩形": "Hollow Rect",
    "椭圆管": "Elliptical Tube", "L形": "L-Shape", "U形(槽钢)": "U-Shape (Channel)",
    "十字形": "Cross-Shape", "T形": "T-Shape", "工字形": "I-Shape"
}

# ==========================================
# 2. 物理引擎核心库 (计算逻辑完全不变)
# ==========================================
class TeachingPhysics:
    @staticmethod
    def calc_section_properties(u, v):
        if u[0] != u[-1] or v[0] != v[-1]:
            u = np.append(u, u[0]); v = np.append(v, v[0])
        a_term = u[:-1] * v[1:] - u[1:] * v[:-1]
        A = 0.5 * np.sum(a_term)
        if abs(A) < 1e-9: return 1.0, 1.0, 1.0
        cu = np.sum((u[:-1] + u[1:]) * a_term) / (6 * A)
        cv = np.sum((v[:-1] + v[1:]) * a_term) / (6 * A)
        u_c = u - cu; v_c = v - cv
        a_term_c = u_c[:-1] * v_c[1:] - u_c[1:] * v_c[:-1]
        I_u = np.sum((v_c[:-1] ** 2 + v_c[:-1] * v_c[1:] + v_c[1:] ** 2) * a_term_c) / 12
        I_v = np.sum((u_c[:-1] ** 2 + u_c[:-1] * u_c[1:] + u_c[1:] ** 2) * a_term_c) / 12
        J = I_u + I_v
        return abs(A), abs(I_u), abs(J)

    @staticmethod
    def generate_component_mesh(L, section_type, dims, deformation_type, load_val, struct_type='梁', factor=1.0):
        a = float(dims.get('a', 0.2)); b = float(dims.get('b', 0.4))
        t = float(dims.get('t', 0.02)); t1 = float(dims.get('t1', 0.02))
        t2 = float(dims.get('t2', 0.02)); t3 = float(dims.get('t3', 0.02))
        aa = float(dims.get('aa', 0.05)); bb = float(dims.get('bb', 0.05))
        eps = 1e-4
        if section_type == '矩形':
            us = np.array([-a / 2, a / 2, a / 2, -a / 2]); vs = np.array([-b / 2, -b / 2, b / 2, b / 2])
        elif section_type == '空心矩形':
            us = np.array([-a / 2, a / 2, a / 2, eps, eps, a / 2 - t, a / 2 - t, -a / 2 + t, -a / 2 + t, -eps, -eps, -a / 2])
            vs = np.array([-b / 2, -b / 2, b / 2, b / 2, b / 2 - t, b / 2 - t, -b / 2 + t, -b / 2 + t, b / 2 - t, b / 2 - t, b / 2, b / 2])
        elif section_type == '椭圆管' or section_type == '圆形':
            theta_out = np.linspace(-np.pi / 2, 3 * np.pi / 2 - 0.1, 30)
            if section_type == '圆形': b = a
            u_out = a / 2 * np.cos(theta_out); v_out = b / 2 * np.sin(theta_out)
            if section_type == '圆形' and 't' not in dims: us, vs = u_out, v_out
            else:
                theta_in = np.linspace(3 * np.pi / 2 - 0.1, -np.pi / 2, 30)
                u_in = (a / 2 - t) * np.cos(theta_in); v_in = (b / 2 - t) * np.sin(theta_in)
                us = np.concatenate([u_out, u_in]); vs = np.concatenate([v_out, v_in])
        elif section_type == 'L形':
            us = np.array([-a / 2, a / 2, a / 2, -a / 2 + t1, -a / 2 + t1, -a / 2])
            vs = np.array([-b / 2, -b / 2, -b / 2 + t2, -b / 2 + t2, b / 2, b / 2])
        elif section_type == 'U形(槽钢)':
            us = np.array([-a / 2, a / 2, a / 2, a / 2 - t2, a / 2 - t2, -a / 2 + t1, -a / 2 + t1, -a / 2])
            vs = np.array([-b / 2, -b / 2, b / 2, b / 2, -b / 2 + t3, -b / 2 + t3, b / 2, b / 2])
        elif section_type == '十字形':
            us = np.array([-aa / 2, aa / 2, aa / 2, a / 2, a / 2, aa / 2, aa / 2, -aa / 2, -aa / 2, -a / 2, -a / 2, -aa / 2])
            vs = np.array([b / 2, b / 2, bb / 2, bb / 2, -bb / 2, -bb / 2, -b / 2, -b / 2, -bb / 2, -bb / 2, bb / 2, bb / 2])
        elif section_type == 'T形':
            us = np.array([-a / 2, a / 2, a / 2, aa / 2, aa / 2, -aa / 2, -aa / 2, -a / 2])
            vs = np.array([-b / 2, -b / 2, -b / 2 + bb, -b / 2 + bb, b / 2, b / 2, -b / 2 + bb, -b / 2 + bb])
        elif section_type == '工字形':
            us = np.array([-a / 2, a / 2, a / 2, bb / 2, bb / 2, aa / 2, aa / 2, -aa / 2, -aa / 2, -bb / 2, -bb / 2, -a / 2])
            vs = np.array([-b / 2, -b / 2, -b / 2 + t2, -b / 2 + t2, b / 2 - t1, b / 2 - t1, b / 2, b / 2, b / 2 - t1, b / 2 - t1, -b / 2 + t2, -b / 2 + t2])
        else:
            us = np.array([-a / 2, a / 2, a / 2, -a / 2]); vs = np.array([-b / 2, -b / 2, b / 2, b / 2])
        
        A, I_u, J = TeachingPhysics.calc_section_properties(us, vs)
        A_ref, I_ref, J_ref = 0.08, 0.001066, 0.0013
        scale_axial = np.clip(A_ref / A, 0.1, 10.0)
        scale_bend = np.clip(I_ref / I_u, 0.1, 20.0)
        scale_twist = np.clip(J_ref / J, 0.1, 20.0)
        
        n_long = 40; long_coords = np.linspace(0, L, n_long)
        nodes = []; faces = []; face_disps = []
        load_eff = load_val * factor
        node_disps = np.zeros((n_long, len(us)))
        
        for i in range(len(long_coords)):
            s = long_coords[i]
            cur_us, cur_vs = us.copy(), vs.copy(); cur_s = np.full_like(us, s)
            if struct_type == '梁':
                if deformation_type == '拉压':
                    strain = load_eff * 0.02 * scale_axial
                    cur_s += strain * (s - L / 2); cur_us *= (1 - 0.3 * strain); cur_vs *= (1 - 0.3 * strain)
                elif deformation_type == '弯曲':
                    k = load_eff * 0.05 * scale_bend; deflect = k * ((s - L / 2) / L) ** 2 * L
                    cur_vs += deflect; theta = 2 * k * ((s - L / 2) / L); cur_s -= vs * np.sin(theta)
                elif deformation_type == '扭转':
                    twist = (load_eff * 1.5 * scale_twist) * ((s - L / 2) / L)
                    u_new = cur_us * np.cos(twist) - cur_vs * np.sin(twist); v_new = cur_us * np.sin(twist) + cur_vs * np.cos(twist)
                    cur_us, cur_vs = u_new, v_new
                elif deformation_type == '剪切':
                    cur_vs += (load_eff * 0.1 * scale_axial) * (s - L / 2)
                disp = np.sqrt((cur_s - s) ** 2 + (cur_us - us) ** 2 + (cur_vs - vs) ** 2)
                for j in range(len(us)): nodes.append([cur_s[j], cur_us[j], cur_vs[j]])
            else:
                if deformation_type == '拉压':
                    strain = load_eff * 0.02 * scale_axial; cur_s += strain * s
                    cur_us *= (1 - 0.3 * strain); cur_vs *= (1 - 0.3 * strain)
                elif deformation_type == '弯曲':
                    k = load_eff * 0.05 * scale_bend; deflect = k * (s / L) ** 2 * L
                    cur_us += deflect; theta = 2 * k * (s / L); cur_s -= us * np.sin(theta)
                elif deformation_type == '扭转':
                    twist = (load_eff * 1.5 * scale_twist) * (s / L)
                    u_new = cur_us * np.cos(twist) - cur_vs * np.sin(twist); v_new = cur_us * np.sin(twist) + cur_vs * np.cos(twist)
                    cur_us, cur_vs = u_new, v_new
                elif deformation_type == '剪切':
                    cur_us += (load_eff * 0.1 * scale_axial) * s
                disp = np.sqrt((cur_s - s) ** 2 + (cur_us - us) ** 2 + (cur_vs - vs) ** 2)
                for j in range(len(us)): nodes.append([cur_us[j], cur_vs[j], cur_s[j]])
            node_disps[i, :] = disp
            
        nodes = np.array(nodes); n_prof = len(us)
        max_disp = np.max(node_disps) if np.max(node_disps) > 1e-9 else 1.0
        
        for i in range(n_long - 1):
            base = i * n_prof; next_base = (i + 1) * n_prof
            for j in range(n_prof):
                j_next = (j + 1) % n_prof
                faces.append([nodes[base + j], nodes[base + j_next], nodes[next_base + j_next], nodes[next_base + j]])
                avg_d = (node_disps[i, j] + node_disps[i, j_next] + node_disps[i + 1, j_next] + node_disps[i + 1, j]) / 4.0
                face_disps.append(avg_d / max_disp)
                
        faces.append([nodes[j] for j in range(n_prof)][::-1])
        face_disps.append(np.mean(node_disps[0, :]) / max_disp)
        faces.append([nodes[(n_long - 1) * n_prof + j] for j in range(n_prof)])
        face_disps.append(np.mean(node_disps[-1, :]) / max_disp)
        
        visuals = []
        if struct_type == '柱': visuals.append({'type': 'fixed_base', 'pos': np.array([0, 0, 0]), 'size': max(a, b) * 2})
        return nodes, faces, face_disps, visuals

# ==========================================
# 3. 绘图函数 (标题和图例全英文，图文不剥离)
# ==========================================
def create_plot(nodes, faces, face_disps, visuals, L, s_type, color_mode, title_en):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor('#f5f5f5')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title_en, pad=15, fontsize=12, fontweight='bold')

    if len(faces) > 0:
        if color_mode == '变形位移云图(按大小着色)':
            cmap = plt.get_cmap('jet')
            colors = cmap(face_disps)
            mesh = Poly3DCollection(faces, alpha=0.9, facecolor=colors, edgecolor='k', linewidths=0.1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label('Displacement (Normalized)', rotation=270, labelpad=15)
        else:
            color = '#dddddd' if 'Initial' in title_en else '#4db8ff'
            mesh = Poly3DCollection(faces, alpha=0.8, facecolor=color, edgecolor='k', linewidths=0.2)
        ax.add_collection3d(mesh)

    for item in visuals:
        if item['type'] == 'fixed_base':
            p = item['pos']; sz = item['size']
            X, Y = np.meshgrid(np.linspace(-sz, sz, 4), np.linspace(-sz, sz, 4))
            Z = np.zeros_like(X) + p[2]
            ax.plot_surface(X, Y, Z, color='#555555', alpha=0.5)

    limit = L * 0.8
    if s_type == '梁':
        ax.set_xlim(-limit * 0.2, L + limit * 0.2)
        ax.set_ylim(-limit / 2, limit / 2)
        ax.set_zlim(-limit / 2, limit / 2)
    else:
        ax.set_xlim(-limit / 2, limit / 2)
        ax.set_ylim(-limit / 2, limit / 2)
        ax.set_zlim(-limit * 0.1, L + limit * 0.1)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=st.session_state.view_angles['elev'], azim=st.session_state.view_angles['azim'])
    return fig

# ==========================================
# 4. Streamlit 侧边栏与交互 UI (保持纯中文)
# ==========================================
st.sidebar.title("参数配置")
model_type = st.sidebar.radio("选择模型类型", ["水平梁 (两端无约束)", "竖直柱 (底部固定)"])
struct_type = "梁" if "梁" in model_type else "柱"

st.sidebar.header("1. 建立几何模型")
L = st.sidebar.number_input("构件长度 L (m)", min_value=0.1, value=5.0, step=0.5)
sec_configs = {
    "矩形": {"a": "0.2", "b": "0.4"}, "圆形": {"a(D)": "0.3"},
    "空心矩形": {"a": "0.2", "b": "0.4", "t": "0.02"},
    "椭圆管": {"a": "0.2", "b": "0.4", "t": "0.02"},
    "L形": {"a": "0.2", "b": "0.3", "t1(aa)": "0.02", "t2(bb)": "0.02"},
    "U形(槽钢)": {"a": "0.2", "b": "0.3", "t1": "0.02", "t2": "0.02", "t3": "0.02"},
    "十字形": {"a": "0.3", "b": "0.3", "aa": "0.05", "bb": "0.05"},
    "T形": {"a": "0.3", "b": "0.3", "aa": "0.05", "bb": "0.05"},
    "工字形": {"a": "0.2", "b": "0.4", "aa": "0.2", "bb": "0.02", "t1": "0.02", "t2": "0.02"}
}
sec_type = st.sidebar.selectbox("选择截面类型", list(sec_configs.keys()))
dims = {}
config = sec_configs[sec_type]
cols = st.sidebar.columns(2)
for idx, (key, default_val) in enumerate(config.items()):
    clean_key = key.split('(')[0] if '(' in key else key
    with cols[idx % 2]: dims[clean_key] = st.number_input(key, value=float(default_val), format="%.3f")

if st.sidebar.button("显示初始模型", use_container_width=True): st.session_state.action = 'init'

st.sidebar.header("2. 施加荷载与高级渲染")
load_type = st.sidebar.radio("选择受力", ["拉压", "剪切", "弯曲", "扭转"], index=2, horizontal=True)
load_val = st.sidebar.number_input("荷载幅值 (+/-)", value=1.0, step=0.1)
render_mode = st.sidebar.selectbox("渲染模式", ["变形位移云图(按大小着色)", "纯色显示"])

c1, c2 = st.sidebar.columns(2)
with c1: 
    if st.button("计算静态变形", type="primary", use_container_width=True): st.session_state.action = 'calc'
with c2: 
    if st.button("▶ 播放变形动画", use_container_width=True): st.session_state.action = 'anim'

# ==========================================
# 5. 主视窗渲染区
# ==========================================
st.title("3D 梁柱变形分析平台")

v_cols = st.columns([1, 1, 1, 5])
with v_cols[0]:
    if st.button("主视图"): st.session_state.view_angles = {'elev': 20, 'azim': -60}
with v_cols[1]:
    if st.button("俯视图"): st.session_state.view_angles = {'elev': 90, 'azim': -90}
with v_cols[2]:
    if st.button("前视图"): st.session_state.view_angles = {'elev': 0, 'azim': -90}

plot_placeholder = st.empty()

if st.session_state.action is None:
    plot_placeholder.info("👈 请配置左侧参数并点击按钮生成模型")

elif st.session_state.action == 'init':
    nodes, faces, face_disps, visuals = TeachingPhysics.generate_component_mesh(L, sec_type, dims, '无', 0, struct_type, factor=0)
    
    # 转换为英文标题
    en_title = f"{EN_STRUCT[struct_type]} Model (Initial) - {EN_SEC[sec_type]}"
    fig = create_plot(nodes, faces, face_disps, visuals, L, struct_type, color_mode='纯色显示', title_en=en_title)
    plot_placeholder.pyplot(fig)

elif st.session_state.action == 'calc':
    nodes, faces, face_disps, visuals = TeachingPhysics.generate_component_mesh(L, sec_type, dims, load_type, load_val, struct_type, factor=1.0)
    
    # 转换为英文标题
    en_title = f"{EN_STRUCT[struct_type]} Response: {EN_LOAD[load_type]} ({EN_SEC[sec_type]})"
    fig = create_plot(nodes, faces, face_disps, visuals, L, struct_type, color_mode=render_mode, title_en=en_title)
    plot_placeholder.pyplot(fig)

elif st.session_state.action == 'anim':
    anim_steps = 20
    for i in range(anim_steps + 1):
        factor = i / float(anim_steps)
        current_color_mode = '纯色显示' if factor < 0.1 else render_mode
        
        nodes, faces, face_disps, visuals = TeachingPhysics.generate_component_mesh(L, sec_type, dims, load_type, load_val, struct_type, factor=factor)
        
        # 转换为带渲染因子的英文标题
        en_title = f"{EN_STRUCT[struct_type]} Response: {EN_LOAD[load_type]} ({EN_SEC[sec_type]}) [Animating: {factor:.2f}]"
        fig = create_plot(nodes, faces, face_disps, visuals, L, struct_type, color_mode=current_color_mode, title_en=en_title)
        
        plot_placeholder.pyplot(fig)
        plt.close(fig) 
        time.sleep(0.05)
    
    st.session_state.action = 'calc'
