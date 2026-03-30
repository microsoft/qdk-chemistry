# C++ Test Conventions

## Folder Structure

```
tests/
├── data/          # Data structures, containers, serialization
├── algorithms/    # Computational algorithms (SCF, MC, MACIS, etc.)
└── utils/         # Logger, settings, utilities
```

Mirrors `src/qdk/chemistry/`. New tests go in the matching folder.

## Rules

### No qualitative tests
Every test must assert a **quantitative result** or a **specific error**.

- Use `EXPECT_NO_THROW(expr)` — never `try { ... } catch(...) { EXPECT_TRUE(true); }`
- Use `EXPECT_THROW(expr, SpecificExceptionType)` with the exact exception type
- Never write `EXPECT_TRUE(true)` — it passes unconditionally and hides bugs

### Use `TYPED_TEST_SUITE` for type-parameterized tests
When multiple container types share an interface (e.g., CAS/SCI wavefunction containers), use `TYPED_TEST_SUITE` with a traits class. **Never copy-paste a test file and change the type name.**

### Use fixtures for repeated setup
If you see the same 10+ line setup block in multiple tests, extract it into a test fixture's `SetUp()` or a helper method.

### One serialization roundtrip per format
Test JSON and HDF5 roundtrips once per data class with representative data. Don't write separate roundtrip tests for every field combination — that tests the serialization library, not the domain logic.

### Don't test language features
Don't test that `Element::H == Element::H` or that an enum has specific values. These test C++ itself, not your code.

### Naming
- Fixture: `{Component}Test` (e.g., `StructureBasicTest`, `MacisAsciTest`)
- Test: `TEST_F(FixtureName, DescriptiveBehavior)` (e.g., `TEST_F(ScfTest, WaterDef2SvpEnergy)`)
- Tolerances: centralized in `ut_common.hpp`
