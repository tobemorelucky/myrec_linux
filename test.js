<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>个人简历 - 单页铺满版</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: '微软雅黑', Arial, sans-serif;
        }
        /* 精准匹配A4尺寸，铺满无空隙 */
        body {
            width: 210mm;
            height: 297mm;
            margin: 0 auto;
            padding: 15mm 18mm; /* 适中内边距，保证铺满 */
            background: white;
            overflow: hidden; /* 禁止溢出 */
        }
        /* 头部：照片+基本信息 占比合理 */
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 8mm; /* 增加间距，避免顶部空 */
        }
        .photo {
            width: 40mm;
            height: 55mm;
            margin-right: 10mm;
            border: 1px solid #eee;
        }
        .photo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .basic-info h1 {
            font-size: 22pt;
            margin-bottom: 3mm;
            color: #222;
        }
        .job-intention {
            font-size: 12pt;
            color: #0066cc;
            margin-bottom: 3mm;
            font-weight: 500;
        }
        .contact-info {
            font-size: 10pt;
            line-height: 1.6;
            color: #333;
        }
        /* 模块标题：醒目且占比合理 */
        .section {
            margin-bottom: 6mm; /* 模块间距适中，填充空间 */
        }
        .section-title {
            font-size: 12pt;
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 3mm;
            border-left: 3px solid #0066cc;
            padding-left: 3mm;
            background: #f8f9fc;
            padding: 2mm 3mm;
            border-radius: 0 4px 4px 0;
        }
        /* 内容区：字号/行高适中，铺满页面 */
        .section-content {
            font-size: 9pt; /* 阅读舒适，且填充空间 */
            line-height: 1.5; /* 行高适中，不挤不空 */
            color: #333;
        }
        .experience-item {
            margin-bottom: 4mm; /* 经历项间距，填充空间 */
        }
        .experience-header {
            display: flex;
            justify-content: space-between;
            font-weight: bold;
            margin-bottom: 2mm;
            font-size: 9.5pt;
        }
        .experience-company {
            color: #0066cc;
        }
        .experience-time {
            font-weight: normal;
            color: #666;
            font-size: 9pt;
        }
        /* 列表项：间距适中，填充空间 */
        .experience-desc, .skills-list {
            list-style: disc;
            list-style-position: inside;
            padding-left: 5mm;
            margin-bottom: 2mm;
        }
        .experience-desc li, .skills-list li {
            margin-bottom: 1.5mm; /* 列表项间距，避免空 */
        }
        .edu-course {
            color: #666;
            font-size: 8.5pt;
            margin-left: 5mm;
            margin-top: 1mm;
        }
        /* 打印适配：强制单页，铺满A4 */
        @media print {
            body {
                width: 100%;
                height: 100%;
                padding: 15mm 18mm;
                margin: 0;
                page-break-after: avoid !important;
                page-break-inside: avoid !important;
            }
        }
    </style>
</head>
<body>
    <!-- 头部：照片+基本信息 -->
    <div class="header">
        <div class="photo">
            <img src="https://p11-flow-imagex-download-sign.byteimg.com/tos-cn-i-a9rns2rl98/48b944921ea4424a8c3897694dc86d06.jpg~tplv-a9rns2rl98-24:720:720.image?lk3s=8e244e95&rcl=202603012110041D9E2C272F34C935A461&rrcfp=8a172a1a&x-expires=1772975405&x-signature=GzliwlxXdTv0DijOx9uBra2VxL4%3D" alt="证件照">
        </div>
        <div class="basic-info">
            <h1>个人简历</h1>
            <div class="job-intention">求职意向：行业研究员（消费/宏观政策）/ 资管部助理研究员</div>
            <div class="contact-info">
                性别：男 | 年龄：27岁<br>
                联系电话：17371266549 | 电子邮箱：xlc952@126.com
            </div>
        </div>
    </div>

    <!-- 自我评价 -->
    <div class="section">
        <div class="section-title">自我评价</div>
        <div class="section-content">
            具备985本科+财会硕士复合背景，拥有3年大型集团及医药/农业实业财务实操经验，对二级市场报表勾稽、政策边际变化及行业基本面研究有敏锐触觉。熟练操作iFind、Choice及LLM辅助研究工具，抗压能力强，注重细节，追求严谨、高质量的研究成果交付。
        </div>
    </div>

    <!-- 教育背景 -->
    <div class="section">
        <div class="section-title">教育背景</div>
        <div class="section-content">
            <div class="experience-item">
                <div class="experience-header">
                    <span class="experience-company">香港城市大学 | 国际会计(MAIA) | 硕士</span>
                    <span class="experience-time">2024.09 – 2025.10</span>
                </div>
                <div class="edu-course">核心课程：企业治理、国际会计、跨国企业管理、国际财务管理、高级财务报表分析等</div>
            </div>
            <div class="experience-item">
                <div class="experience-header">
                    <span class="experience-company">华中科技大学 | 财务管理 | 本科</span>
                    <span class="experience-time">2016.09 – 2020.06</span>
                </div>
                <div class="edu-course">核心课程：高级财务报表分析、企业估值与并购、金融工程、财务管理、会计学等</div>
            </div>
        </div>
    </div>

    <!-- 实习经历 -->
    <div class="section">
        <div class="section-title">实习经历</div>
        <div class="section-content">
            <div class="experience-item">
                <div class="experience-header">
                    <span class="experience-company">财通证券研究所 | 总量研究（政策）实习生</span>
                    <span class="experience-time">2025.09 – 2026.02</span>
                </div>
                <ul class="experience-desc">
                    <li>宏观政策跟踪：持续追踪国家政府网及各部委政策动向，运用LLM工具对重点政策文件做文本挖掘与语义分析，累计产出周度政策研报二十余份，精准研判行业边际变化趋势。</li>
                    <li>细分赛道覆盖：独立负责服务消费研报冰雪经济模块，通过iFind/Choice金融终端提取近5年产业体量及增速数据，构建CAGR预测模型，从政策红利与消费升级角度切入，完成细分赛道深度研究覆盖。</li>
                    <li>数据建模分析：参与旧城改造、服务消费研报项目的分析框架搭建，系统提取地产、建材、基建等板块样本公司营收、净利润、PE/PB、ROE及产业指数等核心指标，结合趋势分析与回归分析，辅助完成行业景气度研判。</li>
                    <li>投研支持：统筹总量研究组内部周报资料整理，运用金融终端实时监控跨市场高频数据，整理会议纪要及专家访谈摘要，提炼机构投资者观点中的投资增量信息，为投研决策提供支撑。</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- 工作经历 -->
    <div class="section">
        <div class="section-title">工作经历</div>
        <div class="section-content">
            <div class="experience-item">
                <div class="experience-header">
                    <span class="experience-company">湖北龙翔药业科技股份有限公司 | 财务助理</span>
                    <span class="experience-time">2022.02 – 2023.12</span>
                </div>
                <ul class="experience-desc">
                    <li>医药行业研究：结合医药制造企业实务工作，跟踪化学药、医药中间体成本构成及环保政策动向，梳理医药企业财务核算特点与报表逻辑，强化对医疗卫生赛道财务数据真实性的识别与分析能力。</li>
                </ul>
            </div>
            <div class="experience-item">
                <div class="experience-header">
                    <span class="experience-company">海南京粮控股股份有限公司 | 管培生</span>
                    <span class="experience-time">2020.07 – 2021.12</span>
                </div>
                <ul class="experience-desc">
                    <li>供应链轮岗：全流程参与农业/食品板块轮岗工作，覆盖原料采购、仓储物流、终端销售全环节，重点拆解农副产品加工成本结构，建立基于原材料价格及期货价格波动的毛利动态监控机制。</li>
                    <li>经营分析：运用Excel VLOOKUP函数与数据透视表，对各生产部门辅料消耗数据做横纵向对标分析，通过损耗异常预警推动生产工艺改进，有效提升原料利用率与成本控制效率。</li>
                    <li>费用审核与内控：负责天津子公司日常费用审核、金蝶ERP系统录入工作，定期对低值易耗品及生产辅料进行盘点，输出盘点报告与内控建议，确保公司资产安全与会计核算准确性。</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- 项目经历 -->
    <div class="section">
        <div class="section-title">项目经历</div>
        <div class="section-content">
            <div class="experience-item">
                <div class="experience-header">
                    <span class="experience-company">校科创团队 MSE STAR | 费用负责人</span>
                    <span class="experience-time">2018.10 – 2019.09</span>
                </div>
                <ul class="experience-desc">
                    <li>经费管理：作为创始成员主导团队经费管理与制度建设，设计科创经费申请、使用及核销全流程规范，累计申请并管理项目基金百万余元，通过科学的预算分配保障机器人研发项目进度。</li>
                    <li>团队协作：协调电控、视觉、机械组间技术指标落地与资源调配，兼顾团队宣传策划与日常运营管理，助力团队在创立第一年斩获Robomaster中部赛区单项赛二等奖。</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- 技能证书 -->
    <div class="section">
        <div class="section-title">技能证书</div>
        <div class="section-content">
            <ul class="skills-list">
                <li>资质证书：CFA Level I Candidate（预计2026年参加考试）、初级经济师（金融方向）</li>
                <li>语言能力：雅思6.5分（阅读7.0，写作6.0）、大学英语六级（CET-6）</li>
                <li>工具技能：熟练使用iFind/Choice金融终端、Excel（VLOOKUP/数据透视表/建模）、金蝶ERP、LLM辅助研究工具等</li>
            </ul>
        </div>
    </div>
</body>
</html>