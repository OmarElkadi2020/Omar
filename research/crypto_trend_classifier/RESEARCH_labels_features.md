# بحث: أفضل طريقة لليبل الترند وأفضل الفيتشرز لتقليل التأخير (2026-09-25)

## التشخيص: لماذا النموذج بطيء ويتبع الشارت؟
- الليبل الحالي (Oracle/Viterbi بتكلفة تبديل k=1) يعلّم كل شمعة باتجاه **القطعة التي تنتمي إليها**. عند القمة الحقيقية الليبل
  ينقلب فورًا، لكن النموذج السببي لا يرى دليلًا على الانقلاب إلا بعد هبوط كافٍ؛ فيتعلم عمليًا "مُنعِّمًا متأخرًا".
- المقياس المستخدم في التدريب (دقة/MCC لكل شمعة) يعاقب التأخير والإشارة الكاذبة بنفس الوزن، فالنموذج يختار الحذر.
- في الاختبار: النسخة الهادئة تفقد ~41% من حركة كل ترند عند الدخول وتعيد ~42% عند الخروج (missed_move / give-back).
- حدّ نظري: اكتشاف تغيّر الاتجاه مسألة "quickest change detection"؛ CUSUM هو الأمثل رياضيًا (Lorden 1971، Moustakides 1986):
  أقل تأخير ممكن لكل معدل إنذار كاذب. أي نموذج لا يستطيع كسر هذا الحد بنفس المعلومات، لكن يستطيع الاقتراب منه.

## طرق الليبل في الأدبيات
| الطريقة | الفكرة | مع/ضد التأخير |
|---|---|---|
| Fixed horizon | إشارة عائد h شمعة للأمام | ضوضاء عالية، لا يمثل الترند |
| Triple barrier (López de Prado) | أول حاجز يُلمس: ربح / وقف / زمن | ممتاز للصفقات، ليس لحالة الترند |
| **Trend scanning** (López de Prado, MLAM) | لكل شمعة: انحدار للأمام على نوافذ 1..L، اختيار أعلى t-value؛ الإشارة = الاتجاه والقيمة = الثقة | **ليبل للأمام** (الترند القادم لا الحالي) + ثقة تُستخدم كوزن للعينات |
| CTL (Wu et al. 2020) | قطع صعود/هبوط بعتبة ω من القمم/القيعان | يشبه ZigZag، حساس للعتبة |
| **Oracle binary** (Kovačević et al., IEEE Access 2023) | برمجة ديناميكية تعظّم العائد بتكلفة تبديل — الحالي عندنا | الورقة وجدته **الأكثر متانة** أمام أخطاء المصنف |
| **Oracle ternary** (tstrends) | صاعد/محايد/هابط، الانتقال بين الاتجاهين يمر بالمحايد | يعلّم التشوبي كحالة مستقلة بدل إجبارها على اتجاه |
| **Remaining value** (tstrends RemainingValueTuner) | ليبل مستمر = المتبقي من حركة الترند حتى نهايته | يقترب من الصفر **قبل** نهاية الترند ⇒ يعلّم النموذج الخروج مبكرًا |

## الفيتشرز الأقوى في الأدبيات لتقليل التأخير
1. **عوائد متعددة الآفاق مقسومة على التقلب** (1، 21، 63، 126، 252 يوم) — Lim/Wood/Zohren (Deep Momentum Networks).
2. **MACD مُطبَّع بثلاث سرعات** (8/24، 16/48، 32/96) مقسوم على انحراف السعر 63 يوم ثم على انحرافه 252 يوم (Baz et al. 2015).
3. **فيتشرز كشف نقاط التغيّر (Changepoint)**: Wood, Roberts & Zohren 2022 "Slow Momentum with Fast Reversion" أضافوا
   شدة وموقع آخر نقطة تغيّر (GP-CPD) على نوافذ 10–252 يوم ⇒ استجابة أسرع لانقلاب النظام (~+66% أداء 2015–2020).
   بدائل أرخص: إحصاءات CUSUM لتغيّر الانحراف (drift) بعدة حساسيات، Bayesian Online Changepoint (run-length).
4. **التشوبي/المدى**: Choppiness، Efficiency ratio، Variance ratio، R² (أضيفت في المرحلة 10).
5. **الحجم**: OBV و Accumulation/Distribution ضمن الأعلى أهمية في دراسات SHAP على الأسهم.

## خطة الاختبار المقترحة (مرحلة 12)
- نفس بيانات المرحلة 10 (تدريب على أسهم قبل 2011، اختبار على أسهم وكريبتو لم يرها).
- الليبلات المتنافسة: Oracle binary (الحالي)، Oracle ternary، Trend-scanning (إشارة + وزن بالثقة)، Remaining-value (انحدار).
- الفيتشرز: الحالية + MACD المطبَّع + CUSUM/changepoint + OBV/AD.
- **المقياس الأساسي هو ما يشتكي منه المستخدم**: التأخير عند الانقلاب الحقيقي (شموع، ونسبة الحركة الضائعة) عند نفس ميزانية
  الإشارات الكاذبة (≤ 2 انقلاب لكل تغيّر حقيقي)، مع CUSUM الخالص كخط أساس نظري.

## المصادر
- López de Prado, Machine Learning for Asset Managers — trend scanning: https://mlfinpy.readthedocs.io/en/latest/Labelling.html
- Kovačević et al. 2023, Optimal Trend Labeling in Financial Time Series: https://ieeexplore.ieee.org/document/10210534/
- Wu et al. 2020, CTL: https://www.mdpi.com/1099-4300/22/10/1162
- tstrends (Oracle ternary, RemainingValueTuner): https://github.com/agpenas/tstrends
- Wood, Roberts, Zohren 2022: https://github.com/kieranjwood/slow-momentum-fast-reversion
- Quickest change detection (CUSUM optimality): https://arxiv.org/pdf/1210.5552
- Triple barrier + OHLCV (Korea): https://arxiv.org/abs/2504.02249
- Key technical indicators (SHAP): https://www.sciencedirect.com/science/article/pii/S2666827025000143
