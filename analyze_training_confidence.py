"""Анализ уверенности модели на основе логов обучения"""

print("🔍 АНАЛИЗ УВЕРЕННОСТИ МОДЕЛИ (на основе последнего обучения)")
print("="*60)

# Данные из последних логов
entropy = 0.811
predictions = {
    'LONG': 9.3,
    'SHORT': 11.1,
    'FLAT': 79.6
}
true_dist = {
    'LONG': 37.7,
    'SHORT': 37.0,
    'FLAT': 25.4
}

print("\n📊 РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ:")
print(f"LONG:  {predictions['LONG']}% (истинное: {true_dist['LONG']}%)")
print(f"SHORT: {predictions['SHORT']}% (истинное: {true_dist['SHORT']}%)")
print(f"FLAT:  {predictions['FLAT']}% (истинное: {true_dist['FLAT']}%)")

print(f"\n🎲 ЭНТРОПИЯ: {entropy:.3f}")
print("(0 = полная уверенность в одном классе, 1.099 = равномерное распределение)")

# Анализ уверенности на основе распределения
max_pred = max(predictions.values())
min_pred = min(predictions.values())
spread = max_pred - min_pred

print("\n💡 АНАЛИЗ УВЕРЕННОСТИ:")

# Оценка среднего уровня уверенности
# Если модель предсказывает 79.6% FLAT, это означает высокую уверенность в FLAT
avg_confidence_flat = predictions['FLAT'] / 100
avg_confidence_long = predictions['LONG'] / 100 * 3  # Корректировка т.к. мало предсказаний
avg_confidence_short = predictions['SHORT'] / 100 * 3

print(f"\nОценочная средняя уверенность по классам:")
print(f"- При предсказании FLAT: ~{avg_confidence_flat:.1%}")
print(f"- При предсказании LONG: ~{avg_confidence_long:.1%}")
print(f"- При предсказании SHORT: ~{avg_confidence_short:.1%}")

# Проблемы с уверенностью
print("\n⚠️ ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:")

if predictions['FLAT'] > 70:
    print("1. ЧРЕЗМЕРНАЯ УВЕРЕННОСТЬ В FLAT:")
    print(f"   - Модель предсказывает FLAT в {predictions['FLAT']}% случаев")
    print("   - Это указывает на то, что модель слишком уверена в отсутствии движения")
    print("   - Вероятная средняя уверенность в FLAT: >80%")

if predictions['LONG'] < 15 and predictions['SHORT'] < 15:
    print("\n2. НИЗКАЯ УВЕРЕННОСТЬ В ТОРГОВЫХ СИГНАЛАХ:")
    print(f"   - LONG всего {predictions['LONG']}%, SHORT всего {predictions['SHORT']}%")
    print("   - Модель неуверенно предсказывает торговые сигналы")
    print("   - Вероятно, порог уверенности для LONG/SHORT слишком высок")

# Калибровка
calibration_error = abs(predictions['FLAT'] - true_dist['FLAT'])
print(f"\n3. ПЛОХАЯ КАЛИБРОВКА:")
print(f"   - Ошибка калибровки для FLAT: {calibration_error:.1f}%")
print("   - Модель переоценивает вероятность FLAT в 3 раза!")

# Рекомендации
print("\n🎯 РЕКОМЕНДАЦИИ:")

print("\n1. СНИЗИТЬ ПОРОГ УВЕРЕННОСТИ:")
print("   - Текущий confidence_threshold: 0.6 (60%)")
print("   - Рекомендуется снизить до 0.4-0.45")
print("   - Это позволит модели чаще предсказывать LONG/SHORT")

print("\n2. ИСПОЛЬЗОВАТЬ TEMPERATURE SCALING:")
print("   - Добавить температуру T=2.0 для softmax")
print("   - Это сделает распределение вероятностей более равномерным")
print("   - Снизит чрезмерную уверенность в FLAT")

print("\n3. ДОБАВИТЬ ENTROPY REGULARIZATION:")
print("   - Штрафовать слишком низкую энтропию предсказаний")
print("   - Поощрять более равномерное распределение уверенности")

print("\n4. ИЗМЕНИТЬ ИНИЦИАЛИЗАЦИЮ BIAS:")
print("   - Инициализировать bias direction_head с предпочтением LONG/SHORT")
print("   - Это поможет избежать схлопывания в FLAT с самого начала")

# Визуализация
print("\n📊 ВИЗУАЛИЗАЦИЯ ДИСБАЛАНСА:")
print("\nИстинное распределение:")
for cls, pct in true_dist.items():
    bar = '█' * int(pct/2)
    print(f"{cls:5}: {bar} {pct}%")

print("\nПредсказания модели:")
for cls, pct in predictions.items():
    bar = '█' * int(pct/2)
    print(f"{cls:5}: {bar} {pct}%")

print("\n🔴 ВЫВОД: Модель СЛИШКОМ УВЕРЕНА в предсказании FLAT!")
print("Необходимо снизить порог уверенности и применить temperature scaling.")