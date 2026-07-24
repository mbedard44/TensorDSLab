from typing import assert_type

from tensor_dslab import ReadoutConfig
from tensor_dslab.readout.profiles import ds20k_veto


assert_type(ds20k_veto(), ReadoutConfig)
