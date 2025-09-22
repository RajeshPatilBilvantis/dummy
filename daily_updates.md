
# Data Pipeline Documentation

---

## 1. **Pre-Base**

**Logic:**

* Filters client metrics by business, month, and group.
* Calculates **lead values** for the next policy effective date.

**Rules:**

* **CS (Customer Segment):** More than 1 policy.
* **NCS (Non-Customer Segment):** Lead values are null (i.e., less than 1 policy).

---

## 2. **`prod_es_pre-base-axa-party-trn-arc`**

* Derived from **Pre-Base**.
* Adds new columns such as:

  * `age-tier`
  * Policy amount details across rolling windows (1, 3, 6, 12 months)
* Applied for **CS clients**.

---

## 3. **`prod_cs-pre-base-axa-party-em_arc_diff`**

* Based on `prod_es_pre-base-axa-party-trn-arc`.
* Adds **percentage change columns** for metrics such as:

  * AUM
  * Face amount
  * Other policy metrics
* **Specific to CS clients.**

---

## 4. **`prod_non-cs-pre-base-axa-party-em-arc` / 
## 5. **`prod_non-cs-pre-base-axa-party-trn-arc-diff`**

* Equivalent transformations as in CS pipeline, but applied to **NCS clients**.
* Includes:

  * Policy details over past months
  * Percentage change metrics

