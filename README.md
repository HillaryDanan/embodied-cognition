# Embodied Cognition Gap in Language Models

Testing whether LLMs lack fundamental physical intuitions that human infants possess.

## 🔬 Core Discovery: Stochastic Physics Blindness

**Key Finding**: LLMs exhibit chaotic physics detection (σ = 2.09, range = -1.933 to 7.782), suggesting linguistic pattern matching rather than embodied understanding.

## 📊 Results Summary

- **Detection Rate**: 59.1% (13/22 tests), not significantly above chance (p = 0.124)
- **Coefficient of Variation**: 2.86 (extreme inconsistency)
- **No category differences**: Novel, implicit, and cross-modal tests all equally chaotic (ANOVA p = 0.243)

## 💡 Key Insights

1. **No Coherent Physics Model**: Detection ranges from -1.933 to 7.782 (9.7 point spread)
2. **Lexical Dominance**: Specific words ("mysteriously", "rigid") drive detection, not physics
3. **Context Dependency**: Adding "Naturally," changes gravity detection from -0.448 to 0.623
4. **Universal Failure**: Both obvious (gravity) and subtle (center of mass) violations missed

## 🎯 Implications

- LLMs have statistical correlations where humans have causal models
- Physics "understanding" is actually word association
- Cannot be fixed with more text data - architectural issue
- Variance exceeds mean: classic indicator of no underlying model

## 📁 Repository Structure

```
embodied-cognition/
├── physics_violation_detector.py      # Core detection framework
├── investigate_gravity.py              # Gravity failure analysis
├── novel_physics_tests.py             # Novel scenarios testing
├── implicit_physics_tests.py          # Everyday physics tests
├── crossmodal_physics_tests.py       # Visual->linguistic tests
├── comprehensive_analysis.py          # Statistical meta-analysis
├── lexical_predictor_analysis.py     # Lexical features testing
├── test_other_models.py              # Cross-model validation
└── physics_violations.png            # Visualization of results
```

## 📚 Theoretical Foundation

- **Infant Cognition**: Spelke et al. (1992), Baillargeon (2004) - 6-month-olds detect these violations
- **Embodied Cognition**: Barsalou (2008) - Grounded cognition theory
- **Symbol Grounding**: Harnad (1990) - The symbol grounding problem
- **NLU Limitations**: Bender & Koller (2020) - Climbing towards NLU
- **Force Dynamics**: Talmy (1988) - Force dynamics in language and cognition

## 🚀 Future Directions

1. **Developmental Trajectories**: Compare to 3, 6, 9-month infant data progressions
2. **Causal Interventions**: Test if models understand A→B→C causal chains
3. **Force Dynamics**: More fundamental than object physics (push/pull/resist)
4. **Cross-linguistic**: Do models trained on different languages show different gaps?
5. **Intervention Studies**: Can we teach physics consistency through fine-tuning?
6. **API Model Testing**: Test GPT-4, Claude, Gemini for universality

## 📈 Key Experimental Findings

### Physics Categories Tested
- **Object Permanence**: 80% detection (but inconsistent)
- **Gravity/Support**: Failed spectacularly (-0.506)
- **Novel Physics**: 67% detection, mean -0.272
- **Implicit Physics**: 50% detection, highest variance
- **Cross-modal**: 40% detection, worst performance

### Most Extreme Results
- **Best Detection**: clothing_physics (+7.782)
- **Worst Detection**: center_of_mass (-1.933)
- **Most Puzzling**: gravity violations less surprising than normal physics

## 🔧 Installation & Usage

```bash
# Clone repository
git clone https://github.com/HillaryDanan/embodied-cognition.git
cd embodied-cognition

# Install dependencies
pip install -r requirements.txt

# Run basic physics violation tests
python3 physics_violation_detector.py

# Run comprehensive analysis
python3 comprehensive_analysis.py
```

## 📖 Citation

```bibtex
@misc{embodied-cognition-2025,
  title={Stochastic Physics Blindness: Evidence Against Coherent Physical Reasoning in Language Models},
  author={Danan, Hillary},
  year={2025},
  url={https://github.com/HillaryDanan/embodied-cognition}
}
```

## 🏆 Key Contributions

1. **Identified stochastic (not systematic) physics blindness in LLMs**
2. **Quantified extreme inconsistency (CV = 2.86) in physics detection**
3. **Demonstrated lexical features override physical reasoning**
4. **Established methodology for testing embodied cognition gaps**

## 📝 Status

- ✅ Phase 1: Basic physics violations tested
- ✅ Phase 2: Novel physics scenarios tested  
- ✅ Phase 3: Implicit physics tested
- ✅ Phase 4: Cross-modal tests completed
- ✅ Phase 5: Statistical meta-analysis done
- 🔬 Phase 6: Lexical predictors (in progress)
- 📋 Phase 7: Cross-model validation (planned)
- 📋 Phase 8: API model testing (GPT-4, Claude, Gemini)

## License

MIT

## Contact

Hillary Danan - [GitHub](https://github.com/HillaryDanan)
