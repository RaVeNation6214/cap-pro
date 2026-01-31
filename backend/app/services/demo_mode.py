import logging
import re
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from ..api.schemas import (
    AnalysisResult,
    VulnerabilityPrediction,
    VulnerabilityType,
    LineRisk,
    SampleContract
)
from ..ml.features import FeatureExtractor


@dataclass
class VulnerabilityPattern:
    """Pattern for detecting vulnerabilities."""
    regex: str
    vuln_type: VulnerabilityType
    weight: float
    description: str


class DemoModeAnalyzer:
    """
    Demo mode analyzer that simulates model predictions.
    Uses pattern matching and heuristics to generate realistic predictions.
    """

    # Vulnerability patterns with weights
    PATTERNS = [
        # Reentrancy patterns
        VulnerabilityPattern(
            regex=r'\.call\s*\{?\s*value\s*:',
            vuln_type=VulnerabilityType.REENTRANCY,
            weight=0.85,
            description="External call with value transfer detected"
        ),
        VulnerabilityPattern(
            regex=r'\.call\s*[\(\{].*\n.*=',
            vuln_type=VulnerabilityType.REENTRANCY,
            weight=0.9,
            description="State change after external call (reentrancy pattern)"
        ),

        # Unchecked calls patterns
        VulnerabilityPattern(
            regex=r'\.call\s*[\(\{][^;]*;(?!\s*require)',
            vuln_type=VulnerabilityType.UNCHECKED_CALLS,
            weight=0.8,
            description="External call without return value check"
        ),
        VulnerabilityPattern(
            regex=r'\.send\s*\([^)]*\)\s*;',
            vuln_type=VulnerabilityType.UNCHECKED_CALLS,
            weight=0.75,
            description="send() without return value check"
        ),
        VulnerabilityPattern(
            regex=r'\.delegatecall\s*\(',
            vuln_type=VulnerabilityType.UNCHECKED_CALLS,
            weight=0.85,
            description="delegatecall detected - potential security risk"
        ),

        # Access control patterns
        VulnerabilityPattern(
            regex=r'tx\.origin',
            vuln_type=VulnerabilityType.ACCESS_CONTROL,
            weight=0.95,
            description="tx.origin used for authentication (phishing vulnerability)"
        ),
        VulnerabilityPattern(
            regex=r'require\s*\(\s*tx\.origin',
            vuln_type=VulnerabilityType.ACCESS_CONTROL,
            weight=0.98,
            description="tx.origin in require statement - authentication bypass risk"
        ),

        # Arithmetic patterns
        VulnerabilityPattern(
            regex=r'[\+\-\*](?!\=)(?!.*SafeMath)',
            vuln_type=VulnerabilityType.ARITHMETIC,
            weight=0.3,
            description="Arithmetic operation without SafeMath"
        ),
        VulnerabilityPattern(
            regex=r'\+\+|\-\-',
            vuln_type=VulnerabilityType.ARITHMETIC,
            weight=0.2,
            description="Increment/decrement operation"
        ),
    ]

    # Sample contracts for demo
    SAMPLE_CONTRACTS = [
        SampleContract(
            id="reentrancy_example",
            name="Reentrancy Vulnerable",
            description="Classic reentrancy vulnerability - external call before state update",
            code='''pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance");

        // Vulnerable: external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        // State update after external call - REENTRANCY RISK!
        balances[msg.sender] = 0;
    }

    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}''',
            expected_vulnerabilities=["Reentrancy", "Unchecked Calls"]
        ),
        SampleContract(
            id="tx_origin_example",
            name="tx.origin Authentication",
            description="Uses tx.origin for authentication - vulnerable to phishing",
            code='''pragma solidity ^0.8.0;

contract VulnerableWallet {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    // Vulnerable: using tx.origin for authentication
    function transferTo(address payable _to, uint256 _amount) public {
        require(tx.origin == owner, "Not authorized");
        _to.transfer(_amount);
    }

    function deposit() public payable {}

    receive() external payable {}
}''',
            expected_vulnerabilities=["Access Control"]
        ),
        SampleContract(
            id="unchecked_call_example",
            name="Unchecked External Call",
            description="External call return value not checked",
            code='''pragma solidity ^0.8.0;

contract UncheckedCaller {
    function unsafeCall(address target, bytes memory data) public {
        // Vulnerable: return value not checked
        target.call(data);
    }

    function unsafeSend(address payable recipient, uint256 amount) public {
        // Vulnerable: send return value not checked
        recipient.send(amount);
    }

    function unsafeDelegateCall(address target, bytes memory data) public {
        // Vulnerable: delegatecall without checks
        target.delegatecall(data);
    }
}''',
            expected_vulnerabilities=["Unchecked Calls"]
        ),
        SampleContract(
            id="arithmetic_example",
            name="Integer Overflow Risk",
            description="Arithmetic operations without SafeMath (pre-0.8.0 style)",
            code='''pragma solidity ^0.7.0;

contract TokenSale {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    uint256 public price = 1 ether;

    function buy(uint256 amount) public payable {
        // Potential overflow in multiplication
        uint256 cost = amount * price;
        require(msg.value >= cost, "Insufficient payment");

        // Potential overflow in addition
        balances[msg.sender] = balances[msg.sender] + amount;
        totalSupply = totalSupply + amount;
    }

    function transfer(address to, uint256 amount) public {
        // Potential underflow
        balances[msg.sender] = balances[msg.sender] - amount;
        balances[to] = balances[to] + amount;
    }
}''',
            expected_vulnerabilities=["Arithmetic"]
        ),
        SampleContract(
            id="safe_example",
            name="Safe Contract",
            description="Well-written contract following security best practices",
            code='''pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract SafeBank is ReentrancyGuard, Ownable {
    mapping(address => uint256) private balances;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);

    function deposit() public payable {
        require(msg.value > 0, "Must deposit something");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    // Safe: uses ReentrancyGuard and checks-effects-interactions
    function withdraw() public nonReentrant {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance");

        // Effects before interactions
        balances[msg.sender] = 0;

        // Interaction last
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        emit Withdrawal(msg.sender, amount);
    }

    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }
}''',
            expected_vulnerabilities=[]
        ),
        SampleContract(
            id="cross_function_reentrancy",
            name="Cross-Function Reentrancy",
            description="Reentrancy across multiple functions sharing state",
            code='''pragma solidity ^0.8.0;

contract CrossFunctionReentrancy {
    mapping(address => uint256) public balances;
    mapping(address => bool) public hasWithdrawn;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
        hasWithdrawn[msg.sender] = false;
    }

    function withdraw() public {
        require(!hasWithdrawn[msg.sender], "Already withdrawn");
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance");

        // Vulnerable: external call before state updates
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        // Attacker can call transfer() during callback
        balances[msg.sender] = 0;
        hasWithdrawn[msg.sender] = true;
    }

    // This function can be called during reentrancy
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}''',
            expected_vulnerabilities=["Reentrancy"]
        ),
        SampleContract(
            id="delegatecall_vulnerability",
            name="Delegatecall Vulnerability",
            description="Dangerous delegatecall to user-supplied address",
            code='''pragma solidity ^0.8.0;

contract VulnerableProxy {
    address public owner;
    uint256 public funds;

    constructor() {
        owner = msg.sender;
    }

    // Vulnerable: delegatecall to arbitrary address
    function execute(address _target, bytes memory _data) public {
        // No access control!
        // Delegatecall preserves msg.sender and msg.value
        // but executes in context of this contract
        (bool success, ) = _target.delegatecall(_data);
        require(success, "Delegatecall failed");
    }

    function deposit() public payable {
        funds += msg.value;
    }

    function withdraw() public {
        require(msg.sender == owner, "Not owner");
        payable(owner).transfer(funds);
        funds = 0;
    }
}''',
            expected_vulnerabilities=["Unchecked Calls", "Access Control"]
        ),
        SampleContract(
            id="frontrunning_vulnerable",
            name="Frontrunning Vulnerable",
            description="Contract vulnerable to frontrunning attacks",
            code='''pragma solidity ^0.8.0;

contract VulnerableAuction {
    address public highestBidder;
    uint256 public highestBid;
    mapping(address => uint256) public pendingReturns;
    bool public ended;

    function bid() public payable {
        require(!ended, "Auction ended");
        require(msg.value > highestBid, "Bid too low");

        if (highestBidder != address(0)) {
            // Vulnerable: can be frontrun
            pendingReturns[highestBidder] += highestBid;
        }

        highestBidder = msg.sender;
        highestBid = msg.value;
    }

    // Vulnerable: no commit-reveal scheme
    function revealSecretBid(uint256 amount, bytes32 secret) public payable {
        require(keccak256(abi.encodePacked(amount, secret)) != bytes32(0));
        require(msg.value == amount, "Wrong amount");
        // Attacker can see this transaction and frontrun with higher bid
    }

    function withdraw() public {
        uint256 amount = pendingReturns[msg.sender];
        require(amount > 0, "No funds");

        pendingReturns[msg.sender] = 0;
        // Vulnerable: external call
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
    }
}''',
            expected_vulnerabilities=["Reentrancy", "Unchecked Calls"]
        ),
        SampleContract(
            id="dos_vulnerability",
            name="Denial of Service",
            description="Contract vulnerable to DoS via unexpected revert",
            code='''pragma solidity ^0.8.0;

contract VulnerableRefund {
    address[] public contributors;
    mapping(address => uint256) public contributions;
    bool public refundEnabled;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function contribute() public payable {
        require(msg.value > 0, "Must contribute");
        if (contributions[msg.sender] == 0) {
            contributors.push(msg.sender);
        }
        contributions[msg.sender] += msg.value;
    }

    function enableRefund() public {
        require(msg.sender == owner, "Not owner");
        refundEnabled = true;
    }

    // Vulnerable: DoS if one transfer fails
    function refundAll() public {
        require(refundEnabled, "Refund not enabled");

        // If any contributor is a contract that reverts,
        // the entire refund process fails
        for (uint256 i = 0; i < contributors.length; i++) {
            address contributor = contributors[i];
            uint256 amount = contributions[contributor];
            contributions[contributor] = 0;

            // Vulnerable: uses transfer which can revert
            payable(contributor).transfer(amount);
        }
    }
}''',
            expected_vulnerabilities=["Unchecked Calls"]
        ),
        SampleContract(
            id="selfdestruct_vulnerability",
            name="Selfdestruct Vulnerability",
            description="Contract with unprotected selfdestruct",
            code='''pragma solidity ^0.8.0;

contract VulnerableKill {
    address public owner;
    uint256 public balance;

    constructor() {
        owner = msg.sender;
    }

    function deposit() public payable {
        balance += msg.value;
    }

    // Vulnerable: tx.origin check
    function kill() public {
        require(tx.origin == owner, "Not owner");
        selfdestruct(payable(owner));
    }

    // Another vulnerability: anyone can become owner
    function changeOwner(address newOwner) public {
        // Missing access control!
        owner = newOwner;
    }
}''',
            expected_vulnerabilities=["Access Control"]
        ),
        SampleContract(
            id="timestamp_dependence",
            name="Timestamp Dependence",
            description="Contract relying on block.timestamp for critical logic",
            code='''pragma solidity ^0.7.0;

contract VulnerableLottery {
    address public owner;
    address[] public players;
    uint256 public ticketPrice = 0.1 ether;
    uint256 public jackpot;

    constructor() {
        owner = msg.sender;
    }

    function buyTicket() public payable {
        require(msg.value == ticketPrice, "Wrong ticket price");
        players.push(msg.sender);
        jackpot += msg.value;
    }

    // Vulnerable: using block.timestamp for randomness
    function pickWinner() public {
        require(msg.sender == owner, "Not owner");
        require(players.length > 0, "No players");

        // Miners can manipulate block.timestamp!
        uint256 winnerIndex = uint256(
            keccak256(abi.encodePacked(block.timestamp, players.length))
        ) % players.length;

        address winner = players[winnerIndex];

        // Potential underflow in older Solidity
        uint256 prize = jackpot - (jackpot / 10);
        uint256 fee = jackpot / 10;

        jackpot = 0;
        delete players;

        payable(winner).transfer(prize);
        payable(owner).transfer(fee);
    }
}''',
            expected_vulnerabilities=["Arithmetic"]
        ),
        # Real contracts from newALLBUGS dataset
        SampleContract(
            id="dataset_reentrancy_bank",
            name="U_BANK (Real Dataset)",
            description="Real vulnerable contract from newALLBUGS dataset - Classic reentrancy",
            code='''pragma solidity ^0.4.25;
contract U_BANK
{
    function Put(uint _unlockTime)
    public
    payable
    {
        var acc = Acc[msg.sender];
        acc.balance += msg.value;
        acc.unlockTime = _unlockTime>now?_unlockTime:now;
        LogFile.AddMessage(msg.sender,msg.value,"Put");
    }
    function Collect(uint _am)
    public
    payable
    {
        var acc = Acc[msg.sender];
        if( acc.balance>=MinSum && acc.balance>=_am && now>acc.unlockTime){
            if(msg.sender.call.value(_am)()){  // REENTRANCY: external call before state update
                acc.balance-=_am;
                LogFile.AddMessage(msg.sender,_am,"Collect");
            }
        }
    }
    function()
    public
    payable
    {
        Put(0);
    }
    struct Holder
    {
        uint unlockTime;
        uint balance;
    }
    mapping (address => Holder) public Acc;
    Log LogFile;
    uint public MinSum = 2 ether;
    function U_BANK(address log) public{
        LogFile = Log(log);
    }
}
contract Log
{
    struct Message
    {
        address Sender;
        string  Data;
        uint Val;
        uint  Time;
    }
    Message[] public History;
    Message LastMsg;
    function AddMessage(address _adr,uint _val,string _data)
    public
    {
        LastMsg.Sender = _adr;
        LastMsg.Time = now;
        LastMsg.Val = _val;
        LastMsg.Data = _data;
        History.push(LastMsg);
    }
}''',
            expected_vulnerabilities=["Reentrancy"]
        ),
        SampleContract(
            id="dataset_access_control_bulk",
            name="BulkTransfer (Real Dataset)",
            description="Real vulnerable contract from newALLBUGS dataset - tx.origin authentication",
            code='''pragma solidity ^0.4.24;
interface ERC20 {
    function totalSupply() external view returns (uint);
    function balanceOf(address tokenOwner) external view returns (uint balance);
    function transfer(address to, uint tokens) external returns (bool success);
    function transferFrom(address from, address to, uint tokens) external returns (bool success);
}
library SafeMath {
    function add(uint a, uint b) internal pure returns (uint c) {
        c = a + b;
        require(c >= a, "Addition overflow");
    }
    function sub(uint a, uint b) internal pure returns (uint c) {
        require(b <= a, "Subtraction overflow");
        c = a - b;
    }
}
contract BulkTransfer
{
    using SafeMath for uint;
    address owner;
    constructor () public payable {
        owner = msg.sender;
    }
    function multiTransfer(address[] _addresses, uint[] _amounts) public payable returns(bool) {
        uint toReturn = msg.value;
        for (uint i = 0; i < _addresses.length; i++) {
            _safeTransfer(_addresses[i], _amounts[i]);
            toReturn = SafeMath.sub(toReturn, _amounts[i]);
        }
        _safeTransfer(msg.sender, toReturn);
        return true;
    }
    function _safeTransfer(address _to, uint _amount) internal {
        require(_to != 0, "Receipt address can't be 0");
        _to.transfer(_amount);
    }
    function forwardTransaction(address destination, uint amount, uint gasLimit, bytes data) internal {
        require(tx.origin == owner, "Not an administrator");  // ACCESS CONTROL: tx.origin vulnerability
        require(
            destination.call.gas(
                (gasLimit > 0) ? gasLimit : gasleft()
            ).value(amount)(data),
            "operation failed"
        );
    }
}''',
            expected_vulnerabilities=["Access Control"]
        ),
        SampleContract(
            id="dataset_unchecked_call_capsule",
            name="TimeCapsuleEvent (Real Dataset)",
            description="Real vulnerable contract from newALLBUGS dataset - Unchecked send() return",
            code='''pragma solidity ^0.4.17;
contract Ownable {
    address public Owner;
    function Ownable() { Owner = msg.sender; }
    modifier onlyOwner() {
        if( Owner == msg.sender )
            _;
    }
    function transferOwner(address _owner) onlyOwner {
        if( this.balance == 0 ) {
            Owner = _owner;
        }
    }
}
contract TimeCapsuleEvent is Ownable {
    address public Owner;
    mapping (address=>uint) public deposits;
    uint public openDate;
    function initCapsule(uint open) {
        Owner = msg.sender;
        openDate = open;
    }
    function() payable { deposit(); }
    function deposit() payable {
        if( msg.value >= 0.25 ether ) {
            deposits[msg.sender] += msg.value;
        } else throw;
    }
    function withdraw(uint amount) onlyOwner {
        if( now >= openDate ) {
            uint max = deposits[msg.sender];
            if( amount <= max && max > 0 ) {
                msg.sender.send( amount );  // UNCHECKED CALL: return value not checked
            }
        }
    }
    function kill() onlyOwner {
        if( this.balance == 0 )
            suicide( msg.sender );
    }
}''',
            expected_vulnerabilities=["Unchecked Calls"]
        ),
        SampleContract(
            id="dataset_arithmetic_token",
            name="ERC20token (Real Dataset)",
            description="Real vulnerable contract from newALLBUGS dataset - Integer overflow without SafeMath",
            code='''pragma solidity ^0.4.16;
contract ERC20token{
    uint256 public totalSupply;
    string public name;
    uint8 public decimals;
    string public symbol;
    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;
    function ERC20token(uint256 _initialAmount, string _tokenName, uint8 _decimalUnits, string _tokenSymbol) public {
        totalSupply = _initialAmount * 10 ** uint256(_decimalUnits);
        balances[msg.sender] = totalSupply;
        name = _tokenName;
        decimals = _decimalUnits;
        symbol = _tokenSymbol;
    }
    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balances[msg.sender] >= _value && balances[_to] + _value > balances[_to]);
        require(_to != 0x0);
        balances[msg.sender] -= _value;
        balances[_to] += _value;  // ARITHMETIC: potential overflow
        return true;
    }
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(balances[_from] >= _value && allowed[_from][msg.sender] >= _value);
        balances[_to] += _value;  // ARITHMETIC: potential overflow
        balances[_from] -= _value;
        allowed[_from][msg.sender] -= _value;
        return true;
    }
    function balanceOf(address _owner) public constant returns (uint256 balance) {
        return balances[_owner];
    }
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        return true;
    }
    function allowance(address _owner, address _spender) public constant returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }
}''',
            expected_vulnerabilities=["Arithmetic"]
        ),
    ]

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.compiled_patterns = [
            (re.compile(p.regex, re.MULTILINE | re.IGNORECASE), p)
            for p in self.PATTERNS
        ]
        self.logger = logging.getLogger(__name__)

    def analyze(self, code: str) -> AnalysisResult:
        """
        Analyze a smart contract and return vulnerability predictions.
        """
        lines = code.split('\n')
        num_lines = len(lines)
        self.logger.info("Demo analysis start. lines=%s chars=%s", num_lines, len(code))

        # Extract static features
        features = self.feature_extractor.extract(code)
        line_features = self.feature_extractor.extract_line_features(code)
        self.logger.info(
            "Feature extraction complete. tx_origin=%s call_value=%s delegatecall=%s",
            features.has_tx_origin,
            features.has_call_value,
            features.has_delegatecall,
        )

        # Find pattern matches
        vuln_scores: Dict[VulnerabilityType, float] = {
            VulnerabilityType.ARITHMETIC: 0.0,
            VulnerabilityType.ACCESS_CONTROL: 0.0,
            VulnerabilityType.UNCHECKED_CALLS: 0.0,
            VulnerabilityType.REENTRANCY: 0.0,
        }

        affected_lines: Dict[VulnerabilityType, List[int]] = {
            vt: [] for vt in VulnerabilityType
        }

        line_risks: List[LineRisk] = []

        # Check each pattern
        match_counts: Dict[VulnerabilityType, int] = {vt: 0 for vt in VulnerabilityType}
        for compiled_pattern, pattern in self.compiled_patterns:
            for match in compiled_pattern.finditer(code):
                # Find line number
                line_num = code[:match.start()].count('\n') + 1
                match_counts[pattern.vuln_type] += 1

                # Update vulnerability score
                current_score = vuln_scores[pattern.vuln_type]
                new_score = current_score + pattern.weight * (1 - current_score)
                vuln_scores[pattern.vuln_type] = min(new_score, 0.99)

                # Track affected lines
                if line_num not in affected_lines[pattern.vuln_type]:
                    affected_lines[pattern.vuln_type].append(line_num)

        self.logger.info("Pattern scan complete. matches=%s", match_counts)

        # Calculate line-level risks
        for i, line in enumerate(lines):
            line_num = i + 1
            risk_score = 0.0

            # Check if line is affected by any vulnerability
            for vt in VulnerabilityType:
                if line_num in affected_lines[vt]:
                    risk_score = max(risk_score, vuln_scores[vt])

            # Add some risk for suspicious patterns
            if line_features[i]['has_call'] > 0:
                risk_score = max(risk_score, 0.6)
            if line_features[i]['has_tx_origin'] > 0:
                risk_score = max(risk_score, 0.8)
            if line_features[i]['has_delegatecall'] > 0:
                risk_score = max(risk_score, 0.7)

            line_risks.append(LineRisk(
                line_number=line_num,
                content=line,
                risk_score=risk_score,
                is_vulnerable=risk_score > 0.5
            ))

        # Apply feature-based adjustments
        if features.has_tx_origin:
            vuln_scores[VulnerabilityType.ACCESS_CONTROL] = max(
                vuln_scores[VulnerabilityType.ACCESS_CONTROL], 0.85
            )

        if features.has_reentrancy_pattern:
            vuln_scores[VulnerabilityType.REENTRANCY] = max(
                vuln_scores[VulnerabilityType.REENTRANCY], 0.75
            )

        if features.has_unchecked_return:
            vuln_scores[VulnerabilityType.UNCHECKED_CALLS] = max(
                vuln_scores[VulnerabilityType.UNCHECKED_CALLS], 0.7
            )

        # Check for safe patterns that reduce risk
        if 'ReentrancyGuard' in code or 'nonReentrant' in code:
            vuln_scores[VulnerabilityType.REENTRANCY] *= 0.2

        if 'SafeMath' in code or 'pragma solidity ^0.8' in code:
            vuln_scores[VulnerabilityType.ARITHMETIC] *= 0.3

        if 'msg.sender' in code and 'tx.origin' not in code:
            vuln_scores[VulnerabilityType.ACCESS_CONTROL] *= 0.3

        # Add some randomness for realism
        for vt in vuln_scores:
            noise = random.uniform(-0.05, 0.05)
            vuln_scores[vt] = max(0.01, min(0.99, vuln_scores[vt] + noise))

        # Build vulnerability predictions
        vulnerabilities = []
        for vt in VulnerabilityType:
            prob = vuln_scores[vt]
            confidence = self._get_confidence(prob)

            vulnerabilities.append(VulnerabilityPrediction(
                type=vt,
                probability=round(prob, 4),
                confidence=confidence,
                description=self._get_description(vt, prob),
                affected_lines=sorted(affected_lines[vt])
            ))

        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(vuln_scores)
        risk_level = self._get_risk_level(overall_risk)
        self.logger.info("Risk computed. level=%s score=%.4f", risk_level, overall_risk)

        # Generate attention weights (simulated)
        attention_weights = self._generate_attention_weights(lines, line_risks)

        # Generate summary and recommendations
        summary = self._generate_summary(vulnerabilities, overall_risk)
        recommendations = self._generate_recommendations(vulnerabilities)

        result = AnalysisResult(
            overall_risk_score=round(overall_risk, 4),
            risk_level=risk_level,
            vulnerabilities=vulnerabilities,
            line_risks=line_risks,
            attention_weights=attention_weights,
            summary=summary,
            recommendations=recommendations
        )

        self.logger.info(
            "Demo analysis end. vulnerabilities=%s",
            [(v.type, v.probability) for v in vulnerabilities],
        )

        return result

    def _get_confidence(self, probability: float) -> str:
        """Get confidence level from probability."""
        if probability >= 0.8:
            return "High"
        elif probability >= 0.5:
            return "Medium"
        else:
            return "Low"

    def _get_description(self, vuln_type: VulnerabilityType, probability: float) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            VulnerabilityType.ARITHMETIC: (
                "Integer overflow/underflow vulnerabilities occur when arithmetic operations "
                "exceed the maximum or minimum values for the data type, potentially leading "
                "to unexpected behavior or fund theft."
            ),
            VulnerabilityType.ACCESS_CONTROL: (
                "Access control vulnerabilities, particularly those using tx.origin for "
                "authentication, can be exploited through phishing attacks where a malicious "
                "contract tricks users into executing unauthorized transactions."
            ),
            VulnerabilityType.UNCHECKED_CALLS: (
                "Unchecked external call return values can lead to silent failures where "
                "the contract assumes a transfer succeeded when it actually failed, "
                "potentially causing loss of funds or inconsistent state."
            ),
            VulnerabilityType.REENTRANCY: (
                "Reentrancy vulnerabilities occur when external calls are made before "
                "internal state updates, allowing attackers to recursively call back "
                "into the contract and drain funds."
            ),
        }
        return descriptions.get(vuln_type, "Unknown vulnerability type")

    def _calculate_overall_risk(self, scores: Dict[VulnerabilityType, float]) -> float:
        """Calculate overall risk score."""
        # Weighted average with emphasis on high-risk vulnerabilities
        weights = {
            VulnerabilityType.REENTRANCY: 1.0,
            VulnerabilityType.ACCESS_CONTROL: 0.9,
            VulnerabilityType.UNCHECKED_CALLS: 0.8,
            VulnerabilityType.ARITHMETIC: 0.7,
        }

        weighted_sum = sum(scores[vt] * weights[vt] for vt in scores)
        total_weight = sum(weights.values())

        # Also consider max score
        max_score = max(scores.values())

        return 0.6 * (weighted_sum / total_weight) + 0.4 * max_score

    def _get_risk_level(self, score: float) -> str:
        """Get risk level from score."""
        if score >= 0.8:
            return "Critical"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Safe"

    def _generate_attention_weights(
        self,
        lines: List[str],
        line_risks: List[LineRisk]
    ) -> List[float]:
        """Generate simulated attention weights for code windows."""
        # Create windows of 3 lines
        num_windows = max(1, len(lines) // 3)
        attention_weights = []

        for i in range(num_windows):
            start = i * 3
            end = min(start + 3, len(lines))

            # Average risk in this window
            window_risk = sum(
                lr.risk_score for lr in line_risks[start:end]
            ) / max(1, end - start)

            # Add some variation
            weight = window_risk + random.uniform(-0.1, 0.1)
            attention_weights.append(max(0.01, min(1.0, weight)))

        # Normalize
        total = sum(attention_weights)
        if total > 0:
            attention_weights = [w / total for w in attention_weights]

        return attention_weights

    def _generate_summary(
        self,
        vulnerabilities: List[VulnerabilityPrediction],
        overall_risk: float
    ) -> str:
        """Generate analysis summary."""
        high_risk = [v for v in vulnerabilities if v.probability >= 0.7]
        medium_risk = [v for v in vulnerabilities if 0.4 <= v.probability < 0.7]

        if not high_risk and not medium_risk:
            return (
                "This contract appears to be relatively safe. No significant "
                "vulnerabilities were detected. However, always conduct thorough "
                "testing and consider a professional audit for production contracts."
            )

        summary_parts = []

        if high_risk:
            vuln_names = [v.type.value for v in high_risk]
            summary_parts.append(
                f"High-risk vulnerabilities detected: {', '.join(vuln_names)}. "
                "Immediate attention required."
            )

        if medium_risk:
            vuln_names = [v.type.value for v in medium_risk]
            summary_parts.append(
                f"Medium-risk issues found: {', '.join(vuln_names)}. "
                "Review and fix recommended."
            )

        return " ".join(summary_parts)

    def _generate_recommendations(
        self,
        vulnerabilities: List[VulnerabilityPrediction]
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        for vuln in vulnerabilities:
            if vuln.probability < 0.4:
                continue

            if vuln.type == VulnerabilityType.REENTRANCY:
                recommendations.extend([
                    "Use ReentrancyGuard from OpenZeppelin to prevent reentrancy attacks",
                    "Follow the checks-effects-interactions pattern: update state before external calls",
                    "Consider using pull payment pattern instead of push payments"
                ])

            elif vuln.type == VulnerabilityType.ACCESS_CONTROL:
                recommendations.extend([
                    "Replace tx.origin with msg.sender for authentication",
                    "Use OpenZeppelin's Ownable or AccessControl for access management",
                    "Implement multi-signature requirements for critical operations"
                ])

            elif vuln.type == VulnerabilityType.UNCHECKED_CALLS:
                recommendations.extend([
                    "Always check return values of external calls",
                    "Use require() to verify call success: require(success, 'Call failed')",
                    "Consider using OpenZeppelin's Address library for safe calls"
                ])

            elif vuln.type == VulnerabilityType.ARITHMETIC:
                recommendations.extend([
                    "Use Solidity 0.8.0+ which has built-in overflow checking",
                    "For older versions, use SafeMath library from OpenZeppelin",
                    "Consider using unchecked blocks only when overflow is intentional"
                ])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:6]  # Limit to 6 recommendations

    def get_sample_contracts(self) -> List[SampleContract]:
        """Return sample contracts for demo."""
        return self.SAMPLE_CONTRACTS
