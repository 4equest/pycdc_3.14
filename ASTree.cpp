#include <cstring>
#include <cstdint>
#include <cctype>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "ASTree.h"
#include "FastStack.h"
#include "pyc_numeric.h"
#include "bytecode.h"

// This must be a triple quote (''' or """), to handle interpolated string literals containing the opposite quote style.
// E.g. f'''{"interpolated "123' literal"}'''    -> valid.
// E.g. f"""{"interpolated "123' literal"}"""    -> valid.
// E.g. f'{"interpolated "123' literal"}'        -> invalid, unescaped quotes in literal.
// E.g. f'{"interpolated \"123\' literal"}'      -> invalid, f-string expression does not allow backslash.
// NOTE: Nested f-strings not supported.
#define F_STRING_QUOTE "'''"

int g_ast_append_offset_hint = -1;

static void append_to_chain_store(const PycRef<ASTNode>& chainStore,
        PycRef<ASTNode> item, FastStack& stack, const PycRef<ASTBlock>& curblock);
static bool extract_enter_assignment(const PycRef<ASTNode>& node,
        PycRef<PycString>& out_var, PycRef<ASTNode>& out_expr);
static bool is_none_exit_call_any_receiver(const PycRef<ASTNode>& node);
static bool node_uses_name(const PycRef<ASTNode>& node, const char* name);
static bool is_pass_only_node(const PycRef<ASTNode>& node);
static bool is_jump_opcode(int opcode);
static bool is_pass_handler_at_target(const PycRef<PycCode>& code, int target,
        int continuation, int max_stack_depth = 1);
static bool list_contains_offset_in_range(const ASTBlock::list_t& nodes, int start, int end);
static bool is_bare_reraise_except_block(const PycRef<ASTBlock>& blk);
static void normalize_nested_with_except(ASTBlock::list_t& lines);
static void normalize_sequential_try_regions(ASTBlock::list_t& lines);
static void normalize_exception_table_partitions(ASTBlock::list_t& lines);
static void normalize_nested_except_handler_partitions(ASTBlock::list_t& lines);
static void normalize_except_return_rejoin(ASTBlock::list_t& lines);
static bool block_returns_none_only(const PycRef<ASTBlock>& blk);
static bool block_has_terminal_stmt(const PycRef<ASTBlock>& blk);
static bool is_terminal_stmt(const PycRef<ASTNode>& node);
static bool is_bool_constant_node(const PycRef<ASTNode>& node);
static bool is_any_all_builtin_guard(const PycRef<ASTNode>& node);
void print_src(PycRef<ASTNode> node, PycModule* mod, std::ostream& pyc_output);
PycRef<ASTNode> BuildFromCode(PycRef<PycCode> code, PycModule* mod);
static PycRef<ASTNode> try_reconstruct_genexpr_call(const PycRef<ASTNode>& func,
        const ASTCall::pparam_t& pparams, const ASTCall::kwparam_t& kwparams,
        PycModule* mod);
static bool extract_genexpr_components(const PycRef<ASTNode>& container,
        std::vector<PycRef<ASTIterBlock>>& generators, PycRef<ASTNode>& result);

/* Use this to determine if an error occurred (and therefore, if we should
 * avoid cleaning the output tree) */
static bool cleanBuild;

/* Use this to prevent printing return keywords and newlines in lambdas. */
static bool inLambda = false;
/* Use this to suppress synthetic return statements in module code objects. */
static bool inModuleCode = false;
/* Carries context-manager target across nested blocks during print cleanup. */
static PycRef<ASTNode> cleanup_enter_context;
static std::string cleanup_enter_owner_sig;
/* Code object currently being printed, used for post-cleanup heuristics. */
static PycRef<PycCode> cleanup_current_code;
/* Module owning the current code object, for bytecode-aware cleanup heuristics. */
static PycModule* cleanup_current_module = NULL;

/* Use this to keep track of whether we need to print out any docstring and
 * the list of global variables that we are using (such as inside a function). */
static bool printDocstringAndGlobals = false;

/* Use this to keep track of whether we need to print a class or module docstring */
static bool printClassDocstring = true;

// shortcut for all top/pop calls
static PycRef<ASTNode> StackPopTop(FastStack& stack)
{
    const auto node(stack.top());
    stack.pop();
    return node;
}

static PycRef<ASTNode> normalize_slice_bound(PycRef<ASTNode> node)
{
    if (node != NULL
            && node.type() == ASTNode::NODE_OBJECT
            && node.cast<ASTObject>()->object() == Pyc_None) {
        return NULL;
    }
    return node;
}

static PycRef<ASTNode> build_slice_node(PycRef<ASTNode> start,
        PycRef<ASTNode> stop, PycRef<ASTNode> step = {})
{
    start = normalize_slice_bound(std::move(start));
    stop = normalize_slice_bound(std::move(stop));
    step = normalize_slice_bound(std::move(step));

    PycRef<ASTNode> base;
    if (start == NULL && stop == NULL) {
        base = new ASTSlice(ASTSlice::SLICE0);
    } else if (start == NULL) {
        base = new ASTSlice(ASTSlice::SLICE2, start, stop);
    } else if (stop == NULL) {
        base = new ASTSlice(ASTSlice::SLICE1, start, stop);
    } else {
        base = new ASTSlice(ASTSlice::SLICE3, start, stop);
    }

    if (step == NULL)
        return base;

    return new ASTSlice(ASTSlice::SLICE3, base, step);
}

static PycRef<ASTNode> rebuild_constant_slice(const PycRef<ASTNode>& node)
{
    if (node == NULL || node.type() != ASTNode::NODE_OBJECT)
        return node;

    PycRef<PycSlice> slice = node.cast<ASTObject>()->object().try_cast<PycSlice>();
    if (slice == NULL)
        return node;

    PycRef<ASTNode> start;
    if (slice->start() != NULL)
        start = new ASTObject(slice->start());

    PycRef<ASTNode> stop;
    if (slice->stop() != NULL)
        stop = new ASTObject(slice->stop());

    PycRef<ASTNode> step;
    if (slice->step() != NULL)
        step = new ASTObject(slice->step());

    return build_slice_node(start, stop, step);
}

static int node_first_offset(const PycRef<ASTNode>& node)
{
    if (node == NULL || is_pass_only_node(node))
        return -1;
    if (node->offset() >= 0)
        return node->offset();

    PycRef<ASTBlock> blk = node.try_cast<ASTBlock>();
    if (blk == NULL)
        return -1;

    for (const auto& child : blk->nodes()) {
        int off = node_first_offset(child);
        if (off >= 0)
            return off;
    }
    return -1;
}

static int first_depth_zero_protected_start(const PycRef<PycCode>& code)
{
    if (code == NULL)
        return -1;

    int start = std::numeric_limits<int>::max();
    for (const auto& entry : code->exceptionTableEntries()) {
        if (entry.stack_depth == 0)
            start = std::min(start, entry.start_offset);
    }

    return start == std::numeric_limits<int>::max() ? -1 : start;
}

static bool is_python_keyword(const std::string& name)
{
    static const std::unordered_set<std::string> keywords = {
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield"
    };
    return keywords.find(name) != keywords.end();
}

static std::string sanitize_identifier(const std::string& raw)
{
    if (raw.empty())
        return "_";

    // Keep dotted import/module paths untouched (except implicit iterator vars like ".0").
    if (raw.find('.') != std::string::npos && !(raw.size() > 1 && raw[0] == '.'))
        return raw;

    // Preserve Unicode identifiers as-is; byte-wise ctype checks are ASCII-centric.
    for (size_t i = 0; i < raw.size(); ++i) {
        if (static_cast<unsigned char>(raw[i]) >= 0x80)
            return raw;
    }

    bool valid = true;
    unsigned char first = static_cast<unsigned char>(raw[0]);
    if (!(std::isalpha(first) || first == '_'))
        valid = false;

    for (size_t i = 1; i < raw.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(raw[i]);
        if (!(std::isalnum(ch) || ch == '_')) {
            valid = false;
            break;
        }
    }

    if (valid && !is_python_keyword(raw))
        return raw;

    std::string out;
    out.reserve(raw.size() + 1);
    if (!(std::isalpha(first) || first == '_'))
        out.push_back('_');

    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(raw[i]);
        if (std::isalnum(ch) || ch == '_')
            out.push_back(static_cast<char>(ch));
        else
            out.push_back('_');
    }

    if (out.empty())
        out = "_";
    if (is_python_keyword(out))
        out.push_back('_');
    return out;
}

static int next_non_cache_pos(PycBuffer source, PycModule* mod, int pos)
{
    int look_opcode = Pyc::CACHE;
    int look_operand = 0;
    int look_pos = pos;

    while (!source.atEof()) {
        int start = look_pos;
        bc_next(source, mod, look_opcode, look_operand, look_pos);
        if (look_opcode != Pyc::CACHE)
            return start;
    }
    return pos;
}

static int jump_forward_target(PycBuffer source, PycModule* mod, int pos, int operand)
{
    int offs = operand;
    if (mod->verCompare(3, 10) >= 0)
        offs *= sizeof(uint16_t);

    int base = pos;
    if (mod->verCompare(3, 11) >= 0)
        base = next_non_cache_pos(source, mod, pos);
    return base + offs;
}

static int exception_handler_depth_for_target(const PycRef<PycCode>& code, int target)
{
    int depth = -1;
    for (const auto& entry : code->exceptionTableEntries()) {
        if (entry.target != target)
            continue;
        if (entry.stack_depth > depth)
            depth = entry.stack_depth;
    }
    return depth;
}

static int next_meaningful_opcode_pos(const PycRef<PycCode>& code, PycModule* mod, int pos)
{
    if (code == NULL || code->code() == NULL || pos < 0 || pos >= code->code()->length())
        return pos;

    const unsigned char* base = (const unsigned char*)code->code()->value();
    PycBuffer source(base + pos, code->code()->length() - pos);
    int cursor = pos;
    while (!source.atEof()) {
        int look_opcode, look_operand;
        int start = cursor;
        bc_next(source, mod, look_opcode, look_operand, cursor);
        if (look_opcode != Pyc::CACHE && look_opcode != Pyc::NOP)
            return start;
    }
    return pos;
}

static bool is_handler_scan_noop(int opcode)
{
    return opcode == Pyc::CACHE || opcode == Pyc::NOP;
}

static bool is_exception_alias_store_opcode(int opcode)
{
    switch (opcode) {
    case Pyc::STORE_FAST_A:
    case Pyc::STORE_NAME_A:
    case Pyc::STORE_DEREF_A:
    case Pyc::STORE_GLOBAL_A:
        return true;
    default:
        return false;
    }
}

static bool is_exception_alias_delete_opcode(int opcode)
{
    switch (opcode) {
    case Pyc::DELETE_FAST_A:
    case Pyc::DELETE_NAME_A:
    case Pyc::DELETE_DEREF_A:
    case Pyc::DELETE_GLOBAL_A:
        return true;
    default:
        return false;
    }
}

static bool is_none_load_const(const PycRef<PycCode>& code, int opcode, int operand)
{
    if (code == NULL || code->consts() == NULL || opcode != Pyc::LOAD_CONST_A
            || operand < 0 || operand >= code->consts()->size())
        return false;
    PycRef<PycObject> value = code->getConst(operand);
    return value == Pyc_None || value->type() == PycObject::TYPE_NONE;
}

static bool continuation_matches_from(const PycRef<PycCode>& code, int handler_pos,
        int continuation_pos, int handler_limit)
{
    if (code == NULL || cleanup_current_module == NULL || code->code() == NULL)
        return false;

    const int code_len = code->code()->length();
    if (handler_pos < 0 || continuation_pos < 0
            || handler_pos >= code_len || continuation_pos >= code_len)
        return false;

    const unsigned char* base = (const unsigned char*)code->code()->value();
    PycBuffer handler_source(base + handler_pos, code_len - handler_pos);
    PycBuffer continuation_source(base + continuation_pos, code_len - continuation_pos);
    int handler_cursor = handler_pos;
    int continuation_cursor = continuation_pos;
    bool matched_any = false;

    while (!handler_source.atEof() && !continuation_source.atEof() && handler_cursor < handler_limit) {
        int handler_opcode, handler_operand;
        do {
            if (handler_source.atEof() || handler_cursor >= handler_limit)
                return matched_any;
            bc_next(handler_source, cleanup_current_module, handler_opcode, handler_operand, handler_cursor);
        } while (is_handler_scan_noop(handler_opcode));

        int continuation_opcode, continuation_operand;
        do {
            if (continuation_source.atEof())
                return false;
            bc_next(continuation_source, cleanup_current_module,
                    continuation_opcode, continuation_operand, continuation_cursor);
        } while (is_handler_scan_noop(continuation_opcode));

        if (handler_opcode != continuation_opcode || handler_operand != continuation_operand)
            return false;

        matched_any = true;
        if (handler_opcode == Pyc::RETURN_VALUE
                || handler_opcode == Pyc::RETURN_CONST_A
                || handler_opcode == Pyc::INSTRUMENTED_RETURN_VALUE_A
                || handler_opcode == Pyc::INSTRUMENTED_RETURN_CONST_A
                || handler_opcode == Pyc::RERAISE
                || handler_opcode == Pyc::RERAISE_A
                || is_jump_opcode(handler_opcode))
            return true;
    }

    return matched_any && handler_cursor >= handler_limit;
}

static bool is_jump_opcode(int opcode)
{
    switch (opcode) {
    case Pyc::JUMP_FORWARD_A:
    case Pyc::JUMP_BACKWARD_A:
    case Pyc::JUMP_BACKWARD_NO_INTERRUPT_A:
    case Pyc::JUMP_ABSOLUTE_A:
    case Pyc::JUMP_IF_FALSE_A:
    case Pyc::JUMP_IF_TRUE_A:
    case Pyc::JUMP_IF_FALSE_OR_POP_A:
    case Pyc::JUMP_IF_TRUE_OR_POP_A:
    case Pyc::JUMP_IF_NOT_EXC_MATCH_A:
    case Pyc::POP_JUMP_IF_FALSE_A:
    case Pyc::POP_JUMP_IF_TRUE_A:
    case Pyc::POP_JUMP_FORWARD_IF_FALSE_A:
    case Pyc::POP_JUMP_FORWARD_IF_TRUE_A:
    case Pyc::POP_JUMP_FORWARD_IF_NOT_NONE_A:
    case Pyc::POP_JUMP_FORWARD_IF_NONE_A:
    case Pyc::POP_JUMP_BACKWARD_IF_FALSE_A:
    case Pyc::POP_JUMP_BACKWARD_IF_TRUE_A:
    case Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A:
    case Pyc::POP_JUMP_BACKWARD_IF_NONE_A:
    case Pyc::POP_JUMP_IF_NOT_NONE_A:
    case Pyc::POP_JUMP_IF_NONE_A:
    case Pyc::INSTRUMENTED_JUMP_FORWARD_A:
    case Pyc::INSTRUMENTED_JUMP_BACKWARD_A:
    case Pyc::INSTRUMENTED_POP_JUMP_IF_FALSE_A:
    case Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A:
    case Pyc::INSTRUMENTED_POP_JUMP_IF_NOT_NONE_A:
    case Pyc::INSTRUMENTED_POP_JUMP_IF_NONE_A:
        return true;
    default:
        return false;
    }
}

static int except_block_handler_depth(const PycRef<ASTBlock>& block)
{
    if (block == NULL || block->blktype() != ASTBlock::BLK_EXCEPT)
        return -1;
    return block.cast<ASTCondBlock>()->handlerDepth();
}

static PycRef<ASTNode> first_return_in_block(const PycRef<ASTBlock>& block)
{
    if (block == NULL)
        return NULL;
    for (const auto& node : block->nodes()) {
        if (node.try_cast<ASTReturn>() != NULL)
            return node;
    }
    return NULL;
}

static int jump_backward_target(PycBuffer source, PycModule* mod, int pos, int operand)
{
    int offs = operand;
    if (mod->verCompare(3, 10) >= 0)
        offs *= sizeof(uint16_t);

    int base = pos;
    if (mod->verCompare(3, 11) >= 0)
        base = next_non_cache_pos(source, mod, pos);
    return base - offs;
}

/* compiler generates very, VERY similar byte code for if/else statement block and if-expression
 *  statement
 *      if a: b = 1
 *      else: b = 2
 *  expression:
 *      b = 1 if a else 2
 *  (see for instance https://stackoverflow.com/a/52202007)
 *  here, try to guess if just finished else statement is part of if-expression (ternary operator)
 *  if it is, remove statements from the block and put a ternary node on top of stack
 */
static void CheckIfExpr(FastStack& stack, PycRef<ASTBlock> curblock)
{
    if (stack.empty())
        return;
    if (curblock->nodes().size() < 2)
        return;
    auto rit = curblock->nodes().crbegin();
    // the last is "else" block, the one before should be "if" (could be "for", ...)
    if ((*rit)->type() != ASTNode::NODE_BLOCK ||
        (*rit).cast<ASTBlock>()->blktype() != ASTBlock::BLK_ELSE)
        return;
    ++rit;
    if ((*rit)->type() != ASTNode::NODE_BLOCK ||
        (*rit).cast<ASTBlock>()->blktype() != ASTBlock::BLK_IF)
        return;
    auto else_expr = StackPopTop(stack);
    curblock->removeLast();
    auto if_block = curblock->nodes().back();
    auto if_expr = StackPopTop(stack);
    curblock->removeLast();
    if (if_block.type() == ASTNode::NODE_BLOCK
            && is_any_all_builtin_guard(if_block.cast<ASTBlock>().cast<ASTCondBlock>()->cond())
            && is_bool_constant_node(if_expr)) {
        ASTCall::pparam_t params;
        params.push_back(std::move(else_expr));
        stack.push(new ASTCall(if_block.cast<ASTBlock>().cast<ASTCondBlock>()->cond().cast<ASTCompare>()->left(),
                params, ASTCall::kwparam_t()));
        return;
    }
    stack.push(new ASTTernary(std::move(if_block), std::move(if_expr), std::move(else_expr)));
}

static bool is_bool_constant_node(const PycRef<ASTNode>& node)
{
    if (node.type() != ASTNode::NODE_OBJECT)
        return false;

    PycRef<PycObject> obj = node.cast<ASTObject>()->object();
    return obj == Pyc_True || obj == Pyc_False;
}

static bool is_any_all_builtin_guard(const PycRef<ASTNode>& node)
{
    if (node.type() != ASTNode::NODE_COMPARE)
        return false;

    PycRef<ASTCompare> cmp = node.cast<ASTCompare>();
    if (cmp->op() != ASTCompare::CMP_IS)
        return false;
    if (cmp->left().type() != ASTNode::NODE_NAME || cmp->right().type() != ASTNode::NODE_NAME)
        return false;

    PycRef<PycString> left = cmp->left().cast<ASTName>()->name();
    PycRef<PycString> right = cmp->right().cast<ASTName>()->name();
    if (!left->isEqual(right->value()))
        return false;

    return left->isEqual("any") || left->isEqual("all");
}

static bool extract_genexpr_components(const PycRef<ASTNode>& container,
        std::vector<PycRef<ASTIterBlock>>& generators, PycRef<ASTNode>& result)
{
    if (container == NULL)
        return false;

    if (container.type() != ASTNode::NODE_BLOCK && container.type() != ASTNode::NODE_NODELIST)
        return false;

    const ASTNodeList::list_t* nodes = NULL;
    if (container.type() == ASTNode::NODE_BLOCK)
        nodes = &container.cast<ASTBlock>()->nodes();
    else
        nodes = &container.cast<ASTNodeList>()->nodes();

    for (const auto& node : *nodes) {
        if (node.type() == ASTNode::NODE_BLOCK) {
            PycRef<ASTBlock> child = node.cast<ASTBlock>();
            if (child->blktype() == ASTBlock::BLK_FOR || child->blktype() == ASTBlock::BLK_ASYNCFOR) {
                generators.push_back(child.cast<ASTIterBlock>());
                if (extract_genexpr_components(child.cast<ASTNode>(), generators, result))
                    return true;
                generators.pop_back();
            } else if (extract_genexpr_components(child.cast<ASTNode>(), generators, result)) {
                return true;
            }
        } else if (node.type() == ASTNode::NODE_RETURN) {
            PycRef<ASTReturn> ret = node.cast<ASTReturn>();
            if (ret->rettype() == ASTReturn::YIELD) {
                result = ret->value();
                return true;
            }
        }
    }

    return false;
}

static PycRef<ASTNode> try_reconstruct_genexpr_call(const PycRef<ASTNode>& func,
        const ASTCall::pparam_t& pparams, const ASTCall::kwparam_t& kwparams,
        PycModule* mod)
{
    if (func == NULL || func.type() != ASTNode::NODE_FUNCTION || !kwparams.empty() || pparams.empty())
        return NULL;

    PycRef<ASTFunction> fn = func.cast<ASTFunction>();
    PycRef<ASTNode> fn_code = fn->code();
    if (fn_code == NULL || fn_code.type() != ASTNode::NODE_OBJECT)
        return NULL;

    PycRef<PycObject> fn_code_obj = fn_code.cast<ASTObject>()->object();
    if (fn_code_obj->type() != PycObject::TYPE_CODE && fn_code_obj->type() != PycObject::TYPE_CODE2)
        return NULL;

    PycRef<PycCode> code_src = fn_code_obj.cast<PycCode>();
    if (!code_src->name()->isEqual("<genexpr>"))
        return NULL;

    bool savedCleanBuild = cleanBuild;
    PycRef<ASTNode> genexpr_ast = BuildFromCode(code_src, mod);
    bool genexpr_clean = cleanBuild;
    cleanBuild = savedCleanBuild;

    std::vector<PycRef<ASTIterBlock>> generators;
    PycRef<ASTNode> result;
    if (!extract_genexpr_components(genexpr_ast, generators, result)
            || generators.empty() || result == NULL) {
        if (!genexpr_clean)
            cleanBuild = false;
        return NULL;
    }

    PycRef<ASTNode> outer_iter = generators.front()->iter();
    if (outer_iter != NULL && outer_iter.type() == ASTNode::NODE_NAME
            && outer_iter.cast<ASTName>()->name()->isEqual(".0")) {
        PycRef<ASTNode> iter_arg = pparams.front();
        if (iter_arg != NULL && iter_arg.type() == ASTNode::NODE_CALL) {
            PycRef<ASTCall> iter_call = iter_arg.cast<ASTCall>();
            if (iter_call->pparams().empty() && iter_call->kwparams().empty()
                    && !iter_call->hasVar() && !iter_call->hasKW()) {
                iter_arg = iter_call->func();
            }
        }
        generators.front()->setIter(iter_arg);
    }

    PycRef<ASTComprehension> comp = new ASTComprehension(result, ASTComprehension::GENERATOR);
    for (std::vector<PycRef<ASTIterBlock>>::reverse_iterator it = generators.rbegin();
            it != generators.rend(); ++it) {
        comp->addGenerator(*it);
    }

    if (!genexpr_clean)
        cleanBuild = false;
    return comp.cast<ASTNode>();
}

PycRef<ASTNode> BuildFromCode(PycRef<PycCode> code, PycModule* mod)
{
    cleanBuild = true;

    PycBuffer source(code->code()->value(), code->code()->length());

    FastStack stack((mod->majorVer() == 1) ? 20 : code->stackSize());
    stackhist_t stack_hist;

    std::stack<PycRef<ASTBlock> > blocks;
    PycRef<ASTBlock> defblock = new ASTBlock(ASTBlock::BLK_MAIN);
    defblock->init();
    PycRef<ASTBlock> curblock = defblock;
    blocks.push(defblock);

    int opcode, operand;
    int curpos = 0;
    int pos = 0;
    int unpack = 0;
    bool else_pop = false;
    bool need_try = false;
    bool skip_with_except_cleanup = false;
    bool skip_with_except_cleanup_tail = false;
    bool variable_annotations = false;
    std::unordered_map<int, PycRef<ASTNode>> cleared_fast_markers;
    std::unordered_map<int, int> synthetic_loop_heads;

    if (mod->verCompare(3, 8) >= 0) {
        PycBuffer scan_source(code->code()->value(), code->code()->length());
        int scan_opcode, scan_operand;
        int scan_pos = 0;
        while (!scan_source.atEof()) {
            int scan_curpos = scan_pos;
            bc_next(scan_source, mod, scan_opcode, scan_operand, scan_pos);
            if (scan_opcode != Pyc::JUMP_BACKWARD_A)
                continue;

            int target = jump_backward_target(scan_source, mod, scan_pos, scan_operand);
            if (target >= scan_curpos)
                continue;

            int loop_start = next_meaningful_opcode_pos(code, mod, target);
            if (loop_start >= scan_curpos)
                continue;

            int loop_end = next_meaningful_opcode_pos(code, mod, scan_pos);
            int head_opcode = -1;
            int head_operand = 0;
            int head_pos = loop_start;
            PycBuffer head_source((const unsigned char*)code->code()->value() + loop_start,
                    code->code()->length() - loop_start);
            if (!head_source.atEof())
                bc_next(head_source, mod, head_opcode, head_operand, head_pos);
            if (head_opcode == Pyc::FOR_ITER_A || head_opcode == Pyc::INSTRUMENTED_FOR_ITER_A)
                continue;

            bool has_loop_exit_jump = false;
            PycBuffer loop_source((const unsigned char*)code->code()->value() + loop_start,
                    code->code()->length() - loop_start);
            int loop_opcode, loop_operand;
            int loop_pos = loop_start;
            while (!loop_source.atEof() && loop_pos <= scan_curpos) {
                int loop_curpos = loop_pos;
                bc_next(loop_source, mod, loop_opcode, loop_operand, loop_pos);
                if (loop_curpos >= scan_curpos)
                    break;
                if (loop_opcode != Pyc::JUMP_FORWARD_A && loop_opcode != Pyc::INSTRUMENTED_JUMP_FORWARD_A)
                    continue;
                if (jump_forward_target(loop_source, mod, loop_pos, loop_operand) == loop_end) {
                    has_loop_exit_jump = true;
                    break;
                }
            }
            if (!has_loop_exit_jump)
                continue;

            bool has_nested_exception = false;
            for (const auto& entry : code->exceptionTableEntries()) {
                if (entry.start_offset < loop_start || entry.end_offset > loop_end)
                    continue;
                if (entry.target >= loop_end) {
                    has_nested_exception = true;
                    break;
                }
            }
            if (!has_nested_exception)
                continue;

            std::unordered_map<int, int>::iterator it = synthetic_loop_heads.find(loop_start);
            if (it == synthetic_loop_heads.end() || loop_end > it->second)
                synthetic_loop_heads[loop_start] = loop_end;
        }
    }

    auto has_enclosing_while_end = [&](int target) {
        std::stack<PycRef<ASTBlock> > probe = blocks;
        while (!probe.empty()) {
            PycRef<ASTBlock> blk = probe.top();
            probe.pop();
            if (blk != NULL && blk->blktype() == ASTBlock::BLK_WHILE && blk->end() == target)
                return true;
        }
        return false;
    };

#if defined(BLOCK_DEBUG) || defined(STACK_DEBUG)
    const char* code_name = "<unnamed>";
    if (code != NULL && code->name() != NULL)
        code_name = code->name()->value();
    fprintf(stderr, "\n=== BuildFromCode %s firstlineno=%d ===\n", code_name, code->firstLine());
#endif

    while (!source.atEof()) {
#if defined(BLOCK_DEBUG) || defined(STACK_DEBUG)
        fprintf(stderr, "%-7d", pos);
    #ifdef STACK_DEBUG
        fprintf(stderr, "%-5d", (unsigned int)stack_hist.size() + 1);
    #endif
    #ifdef BLOCK_DEBUG
        for (unsigned int i = 0; i < blocks.size(); i++)
            fprintf(stderr, "    ");
        fprintf(stderr, "%s (%d)", curblock->type_str(), curblock->end());
    #endif
        fprintf(stderr, "\n");
#endif

        curpos = pos;
        std::unordered_map<int, int>::const_iterator loop_it = synthetic_loop_heads.find(curpos);
        if (loop_it != synthetic_loop_heads.end()
                && curblock->blktype() != ASTBlock::BLK_WHILE
                && curblock->blktype() != ASTBlock::BLK_FOR) {
            PycRef<ASTBlock> loop_block = new ASTCondBlock(ASTBlock::BLK_WHILE, loop_it->second,
                    new ASTObject(Pyc_True), false);
            loop_block->init();
            blocks.push(loop_block);
            curblock = blocks.top();
        }
        bc_next(source, mod, opcode, operand, pos);
        g_ast_append_offset_hint = curpos;
        if (skip_with_except_cleanup_tail) {
            if (opcode == Pyc::POP_TOP || opcode == Pyc::POP_EXCEPT)
                continue;
            skip_with_except_cleanup_tail = false;
        }

        if (skip_with_except_cleanup) {
            bool end_skip = false;
            if (opcode == Pyc::RERAISE && mod->verCompare(3, 10) < 0) {
                end_skip = true;
            } else if (opcode == Pyc::RERAISE_A
                    && (operand == 1 || (mod->verCompare(3, 14) >= 0 && operand == 2))) {
                end_skip = true;
            }
            if (end_skip) {
                skip_with_except_cleanup = false;
                if (opcode == Pyc::RERAISE_A && mod->verCompare(3, 11) >= 0 && operand == 1)
                    skip_with_except_cleanup_tail = true;
            }
            continue;
        }

        if (need_try && opcode != Pyc::SETUP_EXCEPT_A) {
            need_try = false;

            /* Store the current stack for the except/finally statement(s) */
            stack_hist.push(stack);
            PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, curblock->end(), true);
            blocks.push(tryblock);
            curblock = blocks.top();
        } else if (else_pop
                && opcode != Pyc::JUMP_FORWARD_A
                && opcode != Pyc::JUMP_IF_FALSE_A
                && opcode != Pyc::JUMP_IF_FALSE_OR_POP_A
                && opcode != Pyc::POP_JUMP_IF_FALSE_A
                && opcode != Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                && opcode != Pyc::POP_JUMP_IF_NONE_A
                && opcode != Pyc::POP_JUMP_IF_NOT_NONE_A
                && opcode != Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                && opcode != Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                && opcode != Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                && opcode != Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                && opcode != Pyc::JUMP_IF_TRUE_A
                && opcode != Pyc::JUMP_IF_TRUE_OR_POP_A
                && opcode != Pyc::POP_JUMP_IF_TRUE_A
                && opcode != Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                && opcode != Pyc::INSTRUMENTED_JUMP_FORWARD_A
                && opcode != Pyc::INSTRUMENTED_POP_JUMP_IF_FALSE_A
                && opcode != Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A
                && opcode != Pyc::INSTRUMENTED_POP_JUMP_IF_NONE_A
                && opcode != Pyc::INSTRUMENTED_POP_JUMP_IF_NOT_NONE_A
                && opcode != Pyc::POP_BLOCK) {
            else_pop = false;

            PycRef<ASTBlock> prev = curblock;
            while (prev->end() < pos
                    && prev->blktype() != ASTBlock::BLK_MAIN) {
                if (prev->blktype() != ASTBlock::BLK_CONTAINER) {
                    if (prev->end() == 0) {
                        break;
                    }

                    /* We want to keep the stack the same, but we need to pop
                     * a level off the history. */
                    //stack = stack_hist.top();
                    if (!stack_hist.empty())
                        stack_hist.pop();
                }
                blocks.pop();

                if (blocks.empty())
                    break;

                curblock = blocks.top();
                curblock->append(prev.cast<ASTNode>());

                prev = curblock;

                CheckIfExpr(stack, curblock);
            }
        }

        switch (opcode) {
        case Pyc::BINARY_OP_A:
            {
                if (operand == 26) {
                    /* NB_SUBSCR: subscript operation (Python 3.14+) */
                    PycRef<ASTNode> subscr = stack.top();
                    stack.pop();
                    PycRef<ASTNode> src = stack.top();
                    stack.pop();
                    subscr = rebuild_constant_slice(subscr);
                    stack.push(new ASTSubscr(src, subscr));
                } else {
                    ASTBinary::BinOp op = ASTBinary::from_binary_op(operand);
                    if (op == ASTBinary::BIN_INVALID)
                        fprintf(stderr, "Unsupported `BINARY_OP` operand value: %d\n", operand);
                    PycRef<ASTNode> right = stack.top();
                    stack.pop();
                    PycRef<ASTNode> left = stack.top();
                    stack.pop();
                    stack.push(new ASTBinary(left, right, op));
                }
            }
            break;
        case Pyc::BINARY_ADD:
        case Pyc::BINARY_AND:
        case Pyc::BINARY_DIVIDE:
        case Pyc::BINARY_FLOOR_DIVIDE:
        case Pyc::BINARY_LSHIFT:
        case Pyc::BINARY_MODULO:
        case Pyc::BINARY_MULTIPLY:
        case Pyc::BINARY_OR:
        case Pyc::BINARY_POWER:
        case Pyc::BINARY_RSHIFT:
        case Pyc::BINARY_SUBTRACT:
        case Pyc::BINARY_TRUE_DIVIDE:
        case Pyc::BINARY_XOR:
        case Pyc::BINARY_MATRIX_MULTIPLY:
        case Pyc::INPLACE_ADD:
        case Pyc::INPLACE_AND:
        case Pyc::INPLACE_DIVIDE:
        case Pyc::INPLACE_FLOOR_DIVIDE:
        case Pyc::INPLACE_LSHIFT:
        case Pyc::INPLACE_MODULO:
        case Pyc::INPLACE_MULTIPLY:
        case Pyc::INPLACE_OR:
        case Pyc::INPLACE_POWER:
        case Pyc::INPLACE_RSHIFT:
        case Pyc::INPLACE_SUBTRACT:
        case Pyc::INPLACE_TRUE_DIVIDE:
        case Pyc::INPLACE_XOR:
        case Pyc::INPLACE_MATRIX_MULTIPLY:
            {
                ASTBinary::BinOp op = ASTBinary::from_opcode(opcode);
                if (op == ASTBinary::BIN_INVALID)
                    throw std::runtime_error("Unhandled opcode from ASTBinary::from_opcode");
                PycRef<ASTNode> right = stack.top();
                stack.pop();
                PycRef<ASTNode> left = stack.top();
                stack.pop();
                stack.push(new ASTBinary(left, right, op));
            }
            break;
        case Pyc::BINARY_SUBSCR:
            {
                PycRef<ASTNode> subscr = stack.top();
                stack.pop();
                PycRef<ASTNode> src = stack.top();
                stack.pop();
                stack.push(new ASTSubscr(src, subscr));
            }
            break;
        case Pyc::BREAK_LOOP:
            curblock->append(new ASTKeyword(ASTKeyword::KW_BREAK));
            break;
        case Pyc::BUILD_CLASS:
            {
                PycRef<ASTNode> class_code = stack.top();
                stack.pop();
                PycRef<ASTNode> bases = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();
                stack.push(new ASTClass(class_code, bases, name));
            }
            break;
        case Pyc::BUILD_FUNCTION:
            {
                PycRef<ASTNode> fun_code = stack.top();
                stack.pop();
                stack.push(new ASTFunction(fun_code, {}, {}));
            }
            break;
        case Pyc::BUILD_LIST_A:
            {
                ASTList::value_t values;
                for (int i=0; i<operand; i++) {
                    values.push_front(stack.top());
                    stack.pop();
                }
                stack.push(new ASTList(values));
            }
            break;
        case Pyc::BUILD_SET_A:
            {
                ASTSet::value_t values;
                for (int i=0; i<operand; i++) {
                    values.push_front(stack.top());
                    stack.pop();
                }
                stack.push(new ASTSet(values));
            }
            break;
        case Pyc::BUILD_MAP_A:
            if (mod->verCompare(3, 5) >= 0) {
                auto map = new ASTMap;
                for (int i=0; i<operand; ++i) {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    PycRef<ASTNode> key = stack.top();
                    stack.pop();
                    map->add(key, value);
                }
                stack.push(map);
            } else {
                if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    stack.pop();
                }
                stack.push(new ASTMap());
            }
            break;
        case Pyc::BUILD_CONST_KEY_MAP_A:
            // Top of stack will be a tuple of keys.
            // Values will start at TOS - 1.
            {
                PycRef<ASTNode> keys = stack.top();
                stack.pop();

                ASTConstMap::values_t values;
                values.reserve(operand);
                for (int i = 0; i < operand; ++i) {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    values.push_back(value);
                }

                stack.push(new ASTConstMap(keys, values));
            }
            break;
        case Pyc::STORE_MAP:
            {
                PycRef<ASTNode> key = stack.top();
                stack.pop();
                PycRef<ASTNode> value = stack.top();
                stack.pop();
                PycRef<ASTMap> map = stack.top().cast<ASTMap>();
                map->add(key, value);
            }
            break;
        case Pyc::BUILD_SLICE_A:
            {
                if (operand == 2) {
                    PycRef<ASTNode> end = stack.top();
                    stack.pop();
                    PycRef<ASTNode> start = stack.top();
                    stack.pop();
                    stack.push(build_slice_node(start, end));
                } else if (operand == 3) {
                    PycRef<ASTNode> step = stack.top();
                    stack.pop();
                    PycRef<ASTNode> end = stack.top();
                    stack.pop();
                    PycRef<ASTNode> start = stack.top();
                    stack.pop();
                    stack.push(build_slice_node(start, end, step));
                }
            }
            break;
        case Pyc::BUILD_STRING_A:
            {
                // Nearly identical logic to BUILD_LIST
                ASTList::value_t values;
                for (int i = 0; i < operand; i++) {
                    values.push_front(stack.top());
                    stack.pop();
                }
                stack.push(new ASTJoinedStr(values));
            }
            break;
        case Pyc::BUILD_TUPLE_A:
            {
                // if class is a closure code, ignore this tuple
                PycRef<ASTNode> tos = stack.top();
                if (tos && tos->type() == ASTNode::NODE_LOADBUILDCLASS) {
                    break;
                }

                ASTTuple::value_t values;
                values.resize(operand);
                for (int i=0; i<operand; i++) {
                    values[operand-i-1] = stack.top();
                    stack.pop();
                }
                stack.push(new ASTTuple(values));
            }
            break;
        case Pyc::KW_NAMES_A:
            {

                int kwparams = code->getConst(operand).cast<PycTuple>()->size();
                ASTKwNamesMap kwparamList;
                std::vector<PycRef<PycObject>> keys = code->getConst(operand).cast<PycSimpleSequence>()->values();
                for (int i = 0; i < kwparams; i++) {
                    kwparamList.add(new ASTObject(keys[kwparams - i - 1]), stack.top());
                    stack.pop();
                }
                stack.push(new ASTKwNamesMap(kwparamList));
            }
            break;
        case Pyc::CALL_A:
        case Pyc::CALL_FUNCTION_A:
        case Pyc::INSTRUMENTED_CALL_A:
            {
                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;

                if (mod->verCompare(3, 11) >= 0) {
                    PycRef<ASTNode> object_or_map = stack.top();
                    if (object_or_map.type() == ASTNode::NODE_KW_NAMES_MAP) {
                        stack.pop();
                        PycRef<ASTKwNamesMap> kwparams_map = object_or_map.cast<ASTKwNamesMap>();
                        for (ASTKwNamesMap::map_t::const_iterator it = kwparams_map->values().begin();
                                it != kwparams_map->values().end(); ++it) {
                            kwparamList.push_front(std::make_pair(it->first, it->second));
                            pparams -= 1;
                        }
                    }
                }

                /* Test for the load build class function */
                stack_hist.push(stack);
                int basecnt = 0;
                ASTTuple::value_t bases;
                bases.resize(basecnt);
                PycRef<ASTNode> TOS = stack.top();
                int TOS_type = TOS.type();
                // bases are NODE_NAME and NODE_BINARY at TOS
                while (TOS_type == ASTNode::NODE_NAME || TOS_type == ASTNode::NODE_BINARY) {
                    bases.resize(basecnt + 1);
                    bases[basecnt] = TOS;
                    basecnt++;
                    stack.pop();
                    TOS = stack.top();
                    TOS_type = TOS.type();
                }
                // qualified name is PycString at TOS
                PycRef<ASTNode> name = stack.top();
                stack.pop();
                PycRef<ASTNode> function = stack.top();
                stack.pop();
                PycRef<ASTNode> loadbuild = stack.top();
                stack.pop();

                // Python 3.11+ class build sequence may include PUSH_NULL between
                // LOAD_BUILD_CLASS and MAKE_FUNCTION/CALL.
                if (loadbuild == NULL && !stack.empty()) {
                    PycRef<ASTNode> maybe_loadbuild = stack.top();
                    if (maybe_loadbuild.type() == ASTNode::NODE_LOADBUILDCLASS) {
                        stack.pop();
                        loadbuild = maybe_loadbuild;
                    }
                }

                int loadbuild_type = loadbuild.type();
                if (loadbuild_type == ASTNode::NODE_LOADBUILDCLASS) {
                    PycRef<ASTNode> call = new ASTCall(function, pparamList, kwparamList);
                    stack.push(new ASTClass(call, new ASTTuple(bases), name));
                    stack_hist.pop();
                    break;
                }
                else
                {
                    stack = stack_hist.top();
                    stack_hist.pop();
                }

                /*
                KW_NAMES(i)
                    Stores a reference to co_consts[consti] into an internal variable for use by CALL.
                    co_consts[consti] must be a tuple of strings.
                    New in version 3.11.
                */
                if (mod->verCompare(3, 11) < 0) {
                    for (int i = 0; i < kwparams; i++) {
                        PycRef<ASTNode> val = stack.top();
                        stack.pop();
                        PycRef<ASTNode> key = stack.top();
                        stack.pop();
                        kwparamList.push_front(std::make_pair(key, val));
                    }
                }
                for (int i=0; i<pparams; i++) {
                    PycRef<ASTNode> param = stack.top();
                    stack.pop();
                    if (param.type() == ASTNode::NODE_FUNCTION) {
                        PycRef<ASTNode> fun_code = param.cast<ASTFunction>()->code();
                        PycRef<PycCode> code_src = fun_code.cast<ASTObject>()->object().cast<PycCode>();
                        PycRef<PycString> function_name = code_src->name();
                        if (function_name->isEqual("<lambda>")) {
                            pparamList.push_front(param);
                        } else {
                            // Decorator used
                            PycRef<ASTNode> decor_name = new ASTName(function_name);
                            curblock->append(new ASTStore(param, decor_name));

                            pparamList.push_front(decor_name);
                        }
                    } else {
                        pparamList.push_front(param);
                    }
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();
                if (func == NULL && !stack.empty() && stack.top() != NULL) {
                    func = stack.top();
                    stack.pop();
                }
                if ((opcode == Pyc::CALL_A || opcode == Pyc::INSTRUMENTED_CALL_A) && !stack.empty()) {
                    if (stack.top() == nullptr) {
                        stack.pop();
                    } else if (func.type() == ASTNode::NODE_BINARY
                            && stack.top() == func.cast<ASTBinary>()->left()) {
                        stack.pop();
                    }
                }

                stack.push(new ASTCall(func, pparamList, kwparamList));
            }
            break;
        case Pyc::CALL_KW_A:
        case Pyc::INSTRUMENTED_CALL_KW_A:
            {
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;

                PycRef<ASTNode> kwnames = stack.top();
                stack.pop();

                ASTCall::pparam_t kwnameList;
                if (kwnames && kwnames.type() == ASTNode::NODE_OBJECT) {
                    PycRef<PycObject> kwobj = kwnames.cast<ASTObject>()->object();
                    if (kwobj->type() == PycObject::TYPE_TUPLE
                            || kwobj->type() == PycObject::TYPE_SMALL_TUPLE) {
                        std::vector<PycRef<PycObject>> names = kwobj.cast<PycTuple>()->values();
                        for (std::vector<PycRef<PycObject>>::const_iterator it = names.begin();
                                it != names.end(); ++it) {
                            kwnameList.push_back(new ASTObject(*it));
                        }
                    }
                } else if (kwnames && kwnames.type() == ASTNode::NODE_TUPLE) {
                    ASTTuple::value_t names = kwnames.cast<ASTTuple>()->values();
                    for (ASTTuple::value_t::const_iterator it = names.begin(); it != names.end(); ++it)
                        kwnameList.push_back(*it);
                }

                const int kwparams = static_cast<int>(kwnameList.size());
                for (int i = kwparams - 1; i >= 0; --i) {
                    PycRef<ASTNode> val = stack.top();
                    stack.pop();
                    ASTCall::pparam_t::const_iterator it = kwnameList.begin();
                    std::advance(it, i);
                    kwparamList.push_front(std::make_pair(*it, val));
                }

                int pparams = operand - kwparams;
                if (pparams < 0)
                    pparams = 0;
                for (int i = 0; i < pparams; ++i) {
                    pparamList.push_front(stack.top());
                    stack.pop();
                }

                PycRef<ASTNode> func = stack.top();
                stack.pop();

                if (func == NULL && !stack.empty()
                        && stack.top().type() == ASTNode::NODE_LOADBUILDCLASS
                        && pparamList.size() >= 2) {
                    stack.pop(); // consume LOAD_BUILD_CLASS helper

                    ASTCall::pparam_t::const_iterator pit = pparamList.begin();
                    PycRef<ASTNode> class_func = *pit++;
                    PycRef<ASTNode> class_name = *pit++;

                    ASTTuple::value_t bases;
                    for (; pit != pparamList.end(); ++pit)
                        bases.push_back(*pit);

                    PycRef<ASTNode> class_call = new ASTCall(class_func, ASTCall::pparam_t(), kwparamList);
                    stack.push(new ASTClass(class_call, new ASTTuple(bases), class_name));
                    break;
                }

                if (func == NULL && !stack.empty() && stack.top() != NULL) {
                    func = stack.top();
                    stack.pop();
                }

                if (!stack.empty()) {
                    if (stack.top() == nullptr) {
                        stack.pop();
                    } else if (func.type() == ASTNode::NODE_BINARY
                            && stack.top() == func.cast<ASTBinary>()->left()) {
                        stack.pop();
                    }
                }

                stack.push(new ASTCall(func, pparamList, kwparamList));
            }
            break;
        case Pyc::CALL_FUNCTION_VAR_A:
            {
                PycRef<ASTNode> var = stack.top();
                stack.pop();
                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;
                for (int i=0; i<kwparams; i++) {
                    PycRef<ASTNode> val = stack.top();
                    stack.pop();
                    PycRef<ASTNode> key = stack.top();
                    stack.pop();
                    kwparamList.push_front(std::make_pair(key, val));
                }
                for (int i=0; i<pparams; i++) {
                    pparamList.push_front(stack.top());
                    stack.pop();
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                PycRef<ASTNode> call = new ASTCall(func, pparamList, kwparamList);
                call.cast<ASTCall>()->setVar(var);
                stack.push(call);
            }
            break;
        case Pyc::CALL_FUNCTION_KW_A:
            {
                PycRef<ASTNode> kw = stack.top();
                stack.pop();
                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;
                for (int i=0; i<kwparams; i++) {
                    PycRef<ASTNode> val = stack.top();
                    stack.pop();
                    PycRef<ASTNode> key = stack.top();
                    stack.pop();
                    kwparamList.push_front(std::make_pair(key, val));
                }
                for (int i=0; i<pparams; i++) {
                    pparamList.push_front(stack.top());
                    stack.pop();
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                PycRef<ASTNode> call = new ASTCall(func, pparamList, kwparamList);
                call.cast<ASTCall>()->setKW(kw);
                stack.push(call);
            }
            break;
        case Pyc::CALL_FUNCTION_VAR_KW_A:
            {
                PycRef<ASTNode> kw = stack.top();
                stack.pop();
                PycRef<ASTNode> var = stack.top();
                stack.pop();
                int kwparams = (operand & 0xFF00) >> 8;
                int pparams = (operand & 0xFF);
                ASTCall::kwparam_t kwparamList;
                ASTCall::pparam_t pparamList;
                for (int i=0; i<kwparams; i++) {
                    PycRef<ASTNode> val = stack.top();
                    stack.pop();
                    PycRef<ASTNode> key = stack.top();
                    stack.pop();
                    kwparamList.push_front(std::make_pair(key, val));
                }
                for (int i=0; i<pparams; i++) {
                    pparamList.push_front(stack.top());
                    stack.pop();
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                PycRef<ASTNode> call = new ASTCall(func, pparamList, kwparamList);
                call.cast<ASTCall>()->setKW(kw);
                call.cast<ASTCall>()->setVar(var);
                stack.push(call);
            }
            break;
        case Pyc::CALL_METHOD_A:
            {
                ASTCall::pparam_t pparamList;
                for (int i = 0; i < operand; i++) {
                    PycRef<ASTNode> param = stack.top();
                    stack.pop();
                    if (param.type() == ASTNode::NODE_FUNCTION) {
                        PycRef<ASTNode> fun_code = param.cast<ASTFunction>()->code();
                        PycRef<PycCode> code_src = fun_code.cast<ASTObject>()->object().cast<PycCode>();
                        PycRef<PycString> function_name = code_src->name();
                        if (function_name->isEqual("<lambda>")) {
                            pparamList.push_front(param);
                        } else {
                            // Decorator used
                            PycRef<ASTNode> decor_name = new ASTName(function_name);
                            curblock->append(new ASTStore(param, decor_name));

                            pparamList.push_front(decor_name);
                        }
                    } else {
                        pparamList.push_front(param);
                    }
                }
                PycRef<ASTNode> func = stack.top();
                stack.pop();
                stack.push(new ASTCall(func, pparamList, ASTCall::kwparam_t()));
            }
            break;
        case Pyc::CONTINUE_LOOP_A:
            curblock->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
            break;
        case Pyc::COMPARE_OP_A:
            {
                PycRef<ASTNode> right = stack.top();
                stack.pop();
                PycRef<ASTNode> left = stack.top();
                stack.pop();
                auto arg = operand;
                if (mod->verCompare(3, 12) == 0)
                    arg >>= 4; // changed under GH-100923
                else if (mod->verCompare(3, 13) >= 0)
                    arg >>= 5;
                stack.push(new ASTCompare(left, right, arg));
            }
            break;
        case Pyc::CONTAINS_OP_A:
            {
                PycRef<ASTNode> right = stack.top();
                stack.pop();
                PycRef<ASTNode> left = stack.top();
                stack.pop();
                // The operand will be 0 for 'in' and 1 for 'not in'.
                stack.push(new ASTCompare(left, right, operand ? ASTCompare::CMP_NOT_IN : ASTCompare::CMP_IN));
            }
            break;
        case Pyc::DELETE_ATTR_A:
            {
                PycRef<ASTNode> name = stack.top();
                stack.pop();
                curblock->append(new ASTDelete(new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR)));
            }
            break;
        case Pyc::DELETE_GLOBAL_A:
            code->markGlobal(code->getName(operand));
            /* Fall through */
        case Pyc::DELETE_NAME_A:
            {
                PycRef<PycString> varname = code->getName(operand);

                if (varname->length() >= 2 && varname->value()[0] == '_'
                        && varname->value()[1] == '[') {
                    /* Don't show deletes that are a result of list comps. */
                    break;
                }

                PycRef<ASTNode> name = new ASTName(varname);
                curblock->append(new ASTDelete(name));
            }
            break;
        case Pyc::DELETE_FAST_A:
            {
                PycRef<ASTNode> name;

                if (mod->verCompare(1, 3) < 0)
                    name = new ASTName(code->getName(operand));
                else
                    name = new ASTName(code->getLocal(operand));

                if (name.cast<ASTName>()->name()->value()[0] == '_'
                        && name.cast<ASTName>()->name()->value()[1] == '[') {
                    /* Don't show deletes that are a result of list comps. */
                    break;
                }

                curblock->append(new ASTDelete(name));
            }
            break;
        case Pyc::DELETE_SLICE_0:
            {
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE0))));
            }
            break;
        case Pyc::DELETE_SLICE_1:
            {
                PycRef<ASTNode> upper = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE1, upper))));
            }
            break;
        case Pyc::DELETE_SLICE_2:
            {
                PycRef<ASTNode> lower = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE2, NULL, lower))));
            }
            break;
        case Pyc::DELETE_SLICE_3:
            {
                PycRef<ASTNode> lower = stack.top();
                stack.pop();
                PycRef<ASTNode> upper = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                curblock->append(new ASTDelete(new ASTSubscr(name, new ASTSlice(ASTSlice::SLICE3, upper, lower))));
            }
            break;
        case Pyc::DELETE_SUBSCR:
            {
                PycRef<ASTNode> key = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                curblock->append(new ASTDelete(new ASTSubscr(name, key)));
            }
            break;
        case Pyc::DUP_TOP:
            {
                if (stack.top().type() == PycObject::TYPE_NULL) {
                    stack.push(stack.top());
                } else if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    auto chainstore = stack.top();
                    stack.pop();
                    stack.push(stack.top());
                    stack.push(chainstore);
                } else {
                    stack.push(stack.top());
                    ASTNodeList::list_t targets;
                    stack.push(new ASTChainStore(targets, stack.top()));
                }
            }
            break;
        case Pyc::DUP_TOP_TWO:
            {
                PycRef<ASTNode> first = stack.top();
                stack.pop();
                PycRef<ASTNode> second = stack.top();

                stack.push(first);
                stack.push(second);
                stack.push(first);
            }
            break;
        case Pyc::DUP_TOPX_A:
            {
                std::stack<PycRef<ASTNode> > first;
                std::stack<PycRef<ASTNode> > second;

                for (int i = 0; i < operand; i++) {
                    PycRef<ASTNode> node = stack.top();
                    stack.pop();
                    first.push(node);
                    second.push(node);
                }

                while (first.size()) {
                    stack.push(first.top());
                    first.pop();
                }

                while (second.size()) {
                    stack.push(second.top());
                    second.pop();
                }
            }
            break;
        case Pyc::END_FINALLY:
            {
                bool isFinally = false;
                if (curblock->blktype() == ASTBlock::BLK_FINALLY) {
                    PycRef<ASTBlock> final = curblock;
                    blocks.pop();

                    stack = stack_hist.top();
                    stack_hist.pop();

                    curblock = blocks.top();
                    curblock->append(final.cast<ASTNode>());
                    isFinally = true;
                } else if (curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                    blocks.pop();
                    PycRef<ASTBlock> prev = curblock;

                    bool isUninitAsyncFor = false;
                    if (blocks.top()->blktype() == ASTBlock::BLK_CONTAINER) {
                        auto container = blocks.top();
                        blocks.pop();
                        auto asyncForBlock = blocks.top();
                        isUninitAsyncFor = asyncForBlock->blktype() == ASTBlock::BLK_ASYNCFOR && !asyncForBlock->inited();
                        if (isUninitAsyncFor) {
                            auto tryBlock = container->nodes().front().cast<ASTBlock>();
                            if (!tryBlock->nodes().empty() && tryBlock->blktype() == ASTBlock::BLK_TRY) {
                                auto store = tryBlock->nodes().front().try_cast<ASTStore>();
                                if (store) {
                                    asyncForBlock.cast<ASTIterBlock>()->setIndex(store->dest());
                                }
                            }
                            curblock = blocks.top();
                            stack = stack_hist.top();
                            stack_hist.pop();
                            if (!curblock->inited())
                                fprintf(stderr, "Error when decompiling 'async for'.\n");
                        } else {
                            blocks.push(container);
                        }
                    }

                    if (!isUninitAsyncFor) {
                        if (curblock->size() != 0) {
                            blocks.top()->append(curblock.cast<ASTNode>());
                        }

                        curblock = blocks.top();

                        /* Turn it into an else statement. */
                        if (curblock->end() != pos || curblock.cast<ASTContainerBlock>()->hasFinally()) {
                            PycRef<ASTBlock> elseblk = new ASTBlock(ASTBlock::BLK_ELSE, prev->end());
                            elseblk->init();
                            blocks.push(elseblk);
                            curblock = blocks.top();
                        }
                        else {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                    }
                }

                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    /* This marks the end of the except block(s). */
                    PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                    if (!cont->hasFinally() || isFinally) {
                        /* If there's no finally block, pop the container. */
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(cont.cast<ASTNode>());
                    }
                }
            }
            break;
        case Pyc::EXEC_STMT:
            {
                if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    stack.pop();
                }
                PycRef<ASTNode> loc = stack.top();
                stack.pop();
                PycRef<ASTNode> glob = stack.top();
                stack.pop();
                PycRef<ASTNode> stmt = stack.top();
                stack.pop();

                curblock->append(new ASTExec(stmt, glob, loc));
            }
            break;
        case Pyc::FOR_ITER_A:
        case Pyc::INSTRUMENTED_FOR_ITER_A:
            {
                PycRef<ASTNode> iter = stack.top(); // Iterable
                if (mod->verCompare(3, 12) < 0) {
                    // Do not pop the iterator for py 3.12+
                    stack.pop();
                }
                /* Pop it? Don't pop it? */

                int end;
                bool comprehension = false;

                // before 3.8, there is a SETUP_LOOP instruction with block start and end position,
                //    the operand is usually a jump to a POP_BLOCK instruction
                // after 3.8, block extent has to be inferred implicitly; the operand is a jump to a position after the for block
                if (mod->majorVer() == 3 && mod->minorVer() >= 8) {
                    end = jump_forward_target(source, mod, pos, operand);
                    comprehension = strcmp(code->name()->value(), "<listcomp>") == 0;
                } else {
                    PycRef<ASTBlock> top = blocks.top();
                    end = top->end(); // block end position from SETUP_LOOP
                    if (top->blktype() == ASTBlock::BLK_WHILE) {
                        blocks.pop();
                    } else {
                        comprehension = true;
                    }
                }

                PycRef<ASTIterBlock> forblk = new ASTIterBlock(ASTBlock::BLK_FOR, curpos, end, iter);
                forblk->setComprehension(comprehension);
                blocks.push(forblk.cast<ASTBlock>());
                curblock = blocks.top();

                stack.push(NULL);
            }
            break;
        case Pyc::FOR_LOOP_A:
            {
                PycRef<ASTNode> curidx = stack.top(); // Current index
                stack.pop();
                PycRef<ASTNode> iter = stack.top(); // Iterable
                stack.pop();

                bool comprehension = false;
                PycRef<ASTBlock> top = blocks.top();
                if (top->blktype() == ASTBlock::BLK_WHILE) {
                    blocks.pop();
                } else {
                    comprehension = true;
                }
                PycRef<ASTIterBlock> forblk = new ASTIterBlock(ASTBlock::BLK_FOR, curpos, top->end(), iter);
                forblk->setComprehension(comprehension);
                blocks.push(forblk.cast<ASTBlock>());
                curblock = blocks.top();

                /* Python Docs say:
                      "push the sequence, the incremented counter,
                       and the current item onto the stack." */
                stack.push(iter);
                stack.push(curidx);
                stack.push(NULL); // We can totally hack this >_>
            }
            break;
        case Pyc::GET_AITER:
            {
                // Logic similar to FOR_ITER_A
                PycRef<ASTNode> iter = stack.top(); // Iterable
                stack.pop();

                PycRef<ASTBlock> top = blocks.top();
                if (top->blktype() == ASTBlock::BLK_WHILE) {
                    blocks.pop();
                    PycRef<ASTIterBlock> forblk = new ASTIterBlock(ASTBlock::BLK_ASYNCFOR, curpos, top->end(), iter);
                    blocks.push(forblk.cast<ASTBlock>());
                    curblock = blocks.top();
                    stack.push(nullptr);
                } else {
                     fprintf(stderr, "Unsupported use of GET_AITER outside of SETUP_LOOP\n");
                }
            }
            break;
        case Pyc::GET_ANEXT:
            break;
        case Pyc::FORMAT_VALUE_A:
            {
                auto conversion_flag = static_cast<ASTFormattedValue::ConversionFlag>(operand);
                PycRef<ASTNode> format_spec = nullptr;
                if (conversion_flag & ASTFormattedValue::HAVE_FMT_SPEC) {
                    format_spec = stack.top();
                    stack.pop();
                }
                auto val = stack.top();
                stack.pop();
                stack.push(new ASTFormattedValue(val, conversion_flag, format_spec));
            }
            break;
        case Pyc::FORMAT_SIMPLE:
            {
                PycRef<ASTNode> val = stack.top();
                stack.pop();
                stack.push(new ASTFormattedValue(val, ASTFormattedValue::NONE, nullptr));
            }
            break;
        case Pyc::FORMAT_WITH_SPEC:
            {
                PycRef<ASTNode> format_spec = stack.top();
                stack.pop();
                PycRef<ASTNode> val = stack.top();
                stack.pop();
                auto conversion_flag = static_cast<ASTFormattedValue::ConversionFlag>(ASTFormattedValue::NONE | ASTFormattedValue::HAVE_FMT_SPEC);
                stack.push(new ASTFormattedValue(val, conversion_flag, format_spec));
            }
            break;
        case Pyc::GET_AWAITABLE_A:
        case Pyc::GET_AWAITABLE:
            {
                PycRef<ASTNode> object = stack.top();
                stack.pop();
                stack.push(new ASTAwaitable(object));
            }
            break;
        case Pyc::GET_ITER:
        case Pyc::GET_YIELD_FROM_ITER:
            /* We just entirely ignore this */
            break;
        case Pyc::IMPORT_NAME_A:
            if (mod->majorVer() == 1) {
                stack.push(new ASTImport(new ASTName(code->getName(operand)), NULL));
            } else {
                PycRef<ASTNode> fromlist = stack.top();
                stack.pop();
                if (mod->verCompare(2, 5) >= 0)
                    stack.pop();    // Level -- we don't care
                stack.push(new ASTImport(new ASTName(code->getName(operand)), fromlist));
            }
            break;
        case Pyc::IMPORT_FROM_A:
            stack.push(new ASTName(code->getName(operand)));
            break;
        case Pyc::IMPORT_STAR:
            {
                PycRef<ASTNode> import = stack.top();
                stack.pop();
                curblock->append(new ASTStore(import, NULL));
            }
            break;
        case Pyc::IS_OP_A:
            {
                PycRef<ASTNode> right = stack.top();
                stack.pop();
                PycRef<ASTNode> left = stack.top();
                stack.pop();
                // The operand will be 0 for 'is' and 1 for 'is not'.
                stack.push(new ASTCompare(left, right, operand ? ASTCompare::CMP_IS_NOT : ASTCompare::CMP_IS));
            }
            break;
        case Pyc::CHECK_EXC_MATCH:
            {
                PycRef<ASTNode> right = stack.top();
                stack.pop();
                PycRef<ASTNode> left = stack.top();
                stack.pop();
                stack.push(new ASTCompare(left, right, ASTCompare::CMP_EXCEPTION));
            }
            break;
        case Pyc::MATCH_MAPPING:
        case Pyc::MATCH_SEQUENCE:
            /* Keep stack shape for pattern-matching control flow: these opcodes push a boolean result. */
            stack.push(new ASTObject(Pyc_True));
            break;
        case Pyc::JUMP_IF_FALSE_A:
        case Pyc::JUMP_IF_TRUE_A:
        case Pyc::JUMP_IF_FALSE_OR_POP_A:
        case Pyc::JUMP_IF_TRUE_OR_POP_A:
        case Pyc::POP_JUMP_IF_FALSE_A:
        case Pyc::POP_JUMP_IF_TRUE_A:
        case Pyc::POP_JUMP_IF_NONE_A:
        case Pyc::POP_JUMP_IF_NOT_NONE_A:
        case Pyc::POP_JUMP_BACKWARD_IF_FALSE_A:
        case Pyc::POP_JUMP_BACKWARD_IF_TRUE_A:
        case Pyc::POP_JUMP_BACKWARD_IF_NONE_A:
        case Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A:
        case Pyc::POP_JUMP_FORWARD_IF_FALSE_A:
        case Pyc::POP_JUMP_FORWARD_IF_TRUE_A:
        case Pyc::INSTRUMENTED_POP_JUMP_IF_FALSE_A:
        case Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A:
        case Pyc::INSTRUMENTED_POP_JUMP_IF_NONE_A:
        case Pyc::INSTRUMENTED_POP_JUMP_IF_NOT_NONE_A:
            {
                PycRef<ASTNode> cond = stack.top();
                PycRef<ASTCondBlock> ifblk;
                int popped = ASTCondBlock::UNINITED;

                if (opcode == Pyc::POP_JUMP_IF_FALSE_A
                        || opcode == Pyc::POP_JUMP_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_IF_NONE_A
                        || opcode == Pyc::POP_JUMP_IF_NOT_NONE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_FALSE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_FALSE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_NONE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_NOT_NONE_A) {
                    /* Pop condition before the jump */
                    stack.pop();
                    popped = ASTCondBlock::PRE_POPPED;
                }

                /* Store the current stack for the else statement(s) */
                stack_hist.push(stack);

                if (opcode == Pyc::JUMP_IF_FALSE_OR_POP_A
                        || opcode == Pyc::JUMP_IF_TRUE_OR_POP_A) {
                    /* Pop condition only if condition is met */
                    stack.pop();
                    popped = ASTCondBlock::POPPED;
                }

                const bool jump_if_none = (opcode == Pyc::POP_JUMP_IF_NONE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_NONE_A);
                const bool jump_if_not_none = (opcode == Pyc::POP_JUMP_IF_NOT_NONE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_NOT_NONE_A);
                if (jump_if_none || jump_if_not_none) {
                    // Emit body condition directly (not the jump condition).
                    cond = new ASTCompare(cond, new ASTObject(Pyc_None),
                            jump_if_none ? ASTCompare::CMP_IS_NOT : ASTCompare::CMP_IS);
                }

                /* "Jump if true" means "Jump if not false" */
                bool neg = opcode == Pyc::JUMP_IF_TRUE_A
                        || opcode == Pyc::JUMP_IF_TRUE_OR_POP_A
                        || opcode == Pyc::POP_JUMP_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || opcode == Pyc::INSTRUMENTED_POP_JUMP_IF_TRUE_A;
                if (jump_if_none || jump_if_not_none)
                    neg = false;

                int offs = operand;
                if (mod->verCompare(3, 10) >= 0)
                    offs *= sizeof(uint16_t); // // BPO-27129
                const bool backward_cond_jump = (opcode == Pyc::POP_JUMP_BACKWARD_IF_FALSE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_NONE_A
                        || opcode == Pyc::POP_JUMP_BACKWARD_IF_NOT_NONE_A);
                if (backward_cond_jump) {
                    int rel_base = pos;
                    if (mod->verCompare(3, 11) >= 0)
                        rel_base = next_non_cache_pos(source, mod, pos);
                    offs = rel_base - offs;
                } else if (mod->verCompare(3, 12) >= 0
                        || opcode == Pyc::JUMP_IF_FALSE_A
                        || opcode == Pyc::JUMP_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_TRUE_A
                        || opcode == Pyc::POP_JUMP_FORWARD_IF_FALSE_A) {
                    /* Offset is relative in these cases */
                    int rel_base = pos;
                    if (mod->verCompare(3, 11) >= 0)
                        rel_base = next_non_cache_pos(source, mod, pos);
                    offs += rel_base;
                }

                if (cond.type() == ASTNode::NODE_COMPARE
                        && cond.cast<ASTCompare>()->op() == ASTCompare::CMP_EXCEPTION) {
                    const int dispatch_handler_depth = except_block_handler_depth(curblock);
                    while (curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                        PycRef<ASTCondBlock> except_block = curblock.cast<ASTCondBlock>();
                        if (dispatch_handler_depth >= 0
                                && except_block->handlerDepth() >= 0
                                && except_block->handlerDepth() != dispatch_handler_depth)
                            break;
                        const bool is_dispatch_placeholder = except_block->cond() == NULL;
                        const bool finished_typed_handler = !is_dispatch_placeholder
                                && curblock->end() != 0
                                && curblock->end() <= curpos;
                        if (!is_dispatch_placeholder && !finished_typed_handler) {
                            break;
                        }

                        PycRef<ASTBlock> prev_except = curblock;
                        blocks.pop();
                        curblock = blocks.top();

                        if (is_dispatch_placeholder) {
                            PycRef<ASTNode> preserved_return = first_return_in_block(prev_except);
                            if (preserved_return != NULL)
                                curblock->append(preserved_return);
                        } else if (!is_dispatch_placeholder) {
                            curblock->append(prev_except.cast<ASTNode>());
                        }
                        if (!stack_hist.empty())
                            stack_hist.pop();
                    }

                    int handler_depth = dispatch_handler_depth;
                    if (handler_depth < 0)
                        handler_depth = exception_handler_depth_for_target(code, curpos);
                    ifblk = new ASTCondBlock(ASTBlock::BLK_EXCEPT, offs, cond.cast<ASTCompare>()->right(), false,
                                             handler_depth);
                } else if (curblock->blktype() == ASTBlock::BLK_ELSE
                           && curblock->size() == 0
                           && curblock->end() == offs) {
                    /* Collapse into elif only when the else block contains only the nested if. */
                    blocks.pop();
                    stack = stack_hist.top();
                    stack_hist.pop();
                    ifblk = new ASTCondBlock(ASTBlock::BLK_ELIF, offs, cond, neg);
                } else if (curblock->size() == 0 && !curblock->inited()
                           && curblock->blktype() == ASTBlock::BLK_WHILE) {
                    /* The condition for a while loop */
                    PycRef<ASTBlock> top = blocks.top();
                    blocks.pop();
                    ifblk = new ASTCondBlock(top->blktype(), offs, cond, neg);

                    /* We don't store the stack for loops! Pop it! */
                    stack_hist.pop();
                } else if (curblock->size() == 0 && curblock->end() <= offs
                           && (curblock->blktype() == ASTBlock::BLK_IF
                           || curblock->blktype() == ASTBlock::BLK_ELIF
                           || curblock->blktype() == ASTBlock::BLK_WHILE)) {
                    PycRef<ASTNode> newcond;
                    PycRef<ASTCondBlock> top = curblock.cast<ASTCondBlock>();
                    PycRef<ASTNode> cond1 = top->cond();
                    blocks.pop();

                    if (curblock->blktype() == ASTBlock::BLK_WHILE) {
                        stack_hist.pop();
                    } else {
                        FastStack s_top = stack_hist.top();
                        stack_hist.pop();
                        stack_hist.pop();
                        stack_hist.push(s_top);
                    }

                    if (curblock->end() == offs) {
                        if (top->negative() && neg) {
                            /* Same-target POP_JUMP_IF_TRUE blocks encode an inverted OR guard. */
                            newcond = new ASTBinary(cond1, cond, ASTBinary::BIN_LOG_OR);
                        } else {
                            /* if blah and blah */
                            newcond = new ASTBinary(cond1, cond, ASTBinary::BIN_LOG_AND);
                        }
                    } else if (curblock->end() == curpos && !top->negative()) {
                        /* if blah and blah */
                        newcond = new ASTBinary(cond1, cond, ASTBinary::BIN_LOG_AND);
                    } else {
                        /* if blah or blah */
                        newcond = new ASTBinary(cond1, cond, ASTBinary::BIN_LOG_OR);
                    }
                    ifblk = new ASTCondBlock(top->blktype(), offs, newcond, neg);
                } else if (curblock->blktype() == ASTBlock::BLK_FOR
                            && curblock.cast<ASTIterBlock>()->isComprehension()
                            && mod->verCompare(2, 7) >= 0) {
                    /* Comprehension condition */
                    curblock.cast<ASTIterBlock>()->setCondition(cond);
                    stack_hist.pop();
                    // TODO: Handle older python versions, where condition
                    // is laid out a little differently.
                    break;
                } else {
                    /* Plain old if statement */
                    ifblk = new ASTCondBlock(ASTBlock::BLK_IF, offs, cond, neg);
                }

                if (popped)
                    ifblk->init(popped);

                blocks.push(ifblk.cast<ASTBlock>());
                curblock = blocks.top();
            }
            break;
        case Pyc::JUMP_ABSOLUTE_A:
        // bpo-47120: Replaced JUMP_ABSOLUTE by the relative jump JUMP_BACKWARD.
        case Pyc::JUMP_BACKWARD_A:
        case Pyc::JUMP_BACKWARD_NO_INTERRUPT_A:
            {
                int offs = operand;
                if (opcode == Pyc::JUMP_BACKWARD_A
                        || opcode == Pyc::JUMP_BACKWARD_NO_INTERRUPT_A) {
                    offs = jump_backward_target(source, mod, pos, operand);
                } else if (mod->verCompare(3, 10) >= 0) {
                    offs *= sizeof(uint16_t); // // BPO-27129
                }

                if (offs < pos) {
                    if (curblock->blktype() == ASTBlock::BLK_FOR) {
                        bool is_jump_to_start = offs == curblock.cast<ASTIterBlock>()->start();
                        bool should_pop_for_block = curblock.cast<ASTIterBlock>()->isComprehension();
                        // in v3.8, SETUP_LOOP is deprecated and for blocks aren't terminated by POP_BLOCK, so we add them here
                        bool should_add_for_block = mod->majorVer() == 3 && mod->minorVer() >= 8 && is_jump_to_start && !curblock.cast<ASTIterBlock>()->isComprehension();

                        if (should_pop_for_block || should_add_for_block) {
                            PycRef<ASTNode> top = stack.top();

                            if (top.type() == ASTNode::NODE_COMPREHENSION) {
                                PycRef<ASTComprehension> comp = top.cast<ASTComprehension>();

                                comp->addGenerator(curblock.cast<ASTIterBlock>());
                            }

                            PycRef<ASTBlock> tmp = curblock;
                            blocks.pop();
                            curblock = blocks.top();
                            if (should_add_for_block) {
                                curblock->append(tmp.cast<ASTNode>());
                            }
                        }
                    } else if (curblock->blktype() == ASTBlock::BLK_ELSE) {
                        stack = stack_hist.top();
                        stack_hist.pop();

                        blocks.pop();
                        blocks.top()->append(curblock.cast<ASTNode>());
                        curblock = blocks.top();

                        if (curblock->blktype() == ASTBlock::BLK_CONTAINER
                                && !curblock.cast<ASTContainerBlock>()->hasFinally()) {
                            blocks.pop();
                            blocks.top()->append(curblock.cast<ASTNode>());
                            curblock = blocks.top();
                        }
                    } else if (opcode == Pyc::JUMP_BACKWARD_A
                            && curblock->blktype() == ASTBlock::BLK_WHILE
                            && curblock->end() > 0
                            && curblock->end() <= next_non_cache_pos(source, mod, pos)) {
                    } else {
                        curblock->append(new ASTKeyword(ASTKeyword::KW_CONTINUE));
                    }

                    if (opcode == Pyc::JUMP_BACKWARD_A
                            && curblock->blktype() == ASTBlock::BLK_WHILE
                            && curblock->end() > 0
                            && curblock->end() <= next_non_cache_pos(source, mod, pos)) {
                        PycRef<ASTBlock> done_loop = curblock;
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(done_loop.cast<ASTNode>());
                    }

                    /* We're in a loop, this jumps back to the start */
                    /* I think we'll just ignore this case... */
                    break; // Bad idea? Probably!
                }

                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                    if (cont->hasExcept() && pos < cont->except()) {
                        PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, 0, NULL, false,
                                                                   exception_handler_depth_for_target(code, cont->except()));
                        except->init();
                        blocks.push(except);
                        curblock = blocks.top();
                    }
                    break;
                }

                if (!stack_hist.empty()) {
                    stack = stack_hist.top();
                    stack_hist.pop();
                } else {
                    fprintf(stderr, "Warning: Stack history is empty, something wrong might have happened\n");
                }

                PycRef<ASTBlock> prev = curblock;
                PycRef<ASTBlock> nil;
                bool push = true;

                do {
                    blocks.pop();

                    blocks.top()->append(prev.cast<ASTNode>());

                    if (prev->blktype() == ASTBlock::BLK_IF
                            || prev->blktype() == ASTBlock::BLK_ELIF) {
                        if (push) {
                            stack_hist.push(stack);
                        }
                        PycRef<ASTBlock> next = new ASTBlock(ASTBlock::BLK_ELSE, blocks.top()->end());
                        if (prev->inited() == ASTCondBlock::PRE_POPPED) {
                            next->init(ASTCondBlock::PRE_POPPED);
                        }

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_EXCEPT) {
                        if (push) {
                            stack_hist.push(stack);
                        }
                        PycRef<ASTBlock> next = new ASTCondBlock(ASTBlock::BLK_EXCEPT, blocks.top()->end(), NULL, false,
                                                                 except_block_handler_depth(prev));
                        next->init();

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_ELSE) {
                        /* Special case */
                        prev = blocks.top();
                        if (!push) {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                        push = false;
                    } else {
                        prev = nil;
                    }

                } while (prev != nil);

                curblock = blocks.top();
            }
            break;
        case Pyc::JUMP_FORWARD_A:
        case Pyc::INSTRUMENTED_JUMP_FORWARD_A:
            {
                const int jump_target = jump_forward_target(source, mod, pos, operand);
                const int offs = jump_target - pos;

                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();
                    if (cont->hasExcept()) {
                        stack_hist.push(stack);

                        curblock->setEnd(jump_target);
                        PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, jump_target, NULL, false,
                                                                   exception_handler_depth_for_target(code, cont->except()));
                        except->init();
                        blocks.push(except);
                        curblock = blocks.top();
                    }
                    break;
                }

                if (mod->verCompare(3, 11) >= 0
                        && curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                    PycRef<ASTCondBlock> except_block = curblock.cast<ASTCondBlock>();
                    if (except_block->cond() != NULL
                            && curblock->end() != 0
                            && jump_target < curblock->end()) {
                        bool nested_try_rejoin = false;
                        for (const auto& entry : code->exceptionTableEntries()) {
                            if (entry.stack_depth <= 0 || entry.end_offset != curpos)
                                continue;
                            if (entry.target <= curpos || entry.start_offset >= curblock->end())
                                continue;
                            nested_try_rejoin = true;
                            break;
                        }
                        if (nested_try_rejoin)
                            break;
                    }
                }

                if (!stack_hist.empty()) {
                    if (stack.empty()) // if it's part of if-expression, TOS at the moment is the result of "if" part
                        stack = stack_hist.top();
                    stack_hist.pop();
                }

                if ((curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF)
                        && has_enclosing_while_end(jump_target)) {
                    curblock->append(new ASTKeyword(ASTKeyword::KW_BREAK));
                    PycRef<ASTBlock> done_if = curblock;
                    blocks.pop();
                    if (!blocks.empty()) {
                        curblock = blocks.top();
                        curblock->append(done_if.cast<ASTNode>());
                    }
                    break;
                }

                PycRef<ASTBlock> prev = curblock;
                PycRef<ASTBlock> nil;
                bool push = true;

                do {
                    if (prev->blktype() == ASTBlock::BLK_MAIN) {
                        prev = nil;
                        continue;
                    }
                    blocks.pop();

                    if (!blocks.empty())
                        blocks.top()->append(prev.cast<ASTNode>());

                    if (prev->blktype() == ASTBlock::BLK_IF
                            || prev->blktype() == ASTBlock::BLK_ELIF) {
                        if (offs == 0) {
                            prev = nil;
                            continue;
                        }

                        if (push) {
                            stack_hist.push(stack);
                        }
                        PycRef<ASTBlock> next = new ASTBlock(ASTBlock::BLK_ELSE, jump_target);
                        if (prev->inited() == ASTCondBlock::PRE_POPPED) {
                            next->init(ASTCondBlock::PRE_POPPED);
                        }

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_EXCEPT) {
                        if (offs == 0) {
                            prev = nil;
                            continue;
                        }

                        if (push) {
                            stack_hist.push(stack);
                        }
                        PycRef<ASTBlock> next = new ASTCondBlock(ASTBlock::BLK_EXCEPT, jump_target, NULL, false,
                                                                 except_block_handler_depth(prev));
                        next->init();

                        blocks.push(next.cast<ASTBlock>());
                        prev = nil;
                    } else if (prev->blktype() == ASTBlock::BLK_ELSE) {
                        /* Special case */
                        prev = blocks.top();
                        if (!push) {
                            stack = stack_hist.top();
                            stack_hist.pop();
                        }
                        push = false;

                        if (prev->blktype() == ASTBlock::BLK_MAIN) {
                            /* Something went out of control! */
                            prev = nil;
                        }
                    } else if (prev->blktype() == ASTBlock::BLK_TRY
                            && prev->end() < jump_target) {
                        /* Need to add an except/finally block */
                        stack = stack_hist.top();
                        stack.pop();

                        if (blocks.top()->blktype() == ASTBlock::BLK_CONTAINER) {
                            PycRef<ASTContainerBlock> cont = blocks.top().cast<ASTContainerBlock>();
                            if (cont->hasExcept()) {
                                if (push) {
                                    stack_hist.push(stack);
                                }

                                PycRef<ASTBlock> except = new ASTCondBlock(ASTBlock::BLK_EXCEPT, jump_target, NULL, false,
                                                                            exception_handler_depth_for_target(code, cont->except()));
                                except->init();
                                blocks.push(except);
                            }
                        } else {
                            fprintf(stderr, "AUDIT_WARN_CONTAINER_BLOCK_EXPECTED pos=%d curpos=%d opcode=%s block_type=%d\n",
                                    pos, curpos, Pyc::OpcodeName(opcode), blocks.top()->blktype());
                            cleanBuild = false;
                        }
                        prev = nil;
                    } else {
                        prev = nil;
                    }

                } while (prev != nil);

                if (!blocks.empty()) {
                    curblock = blocks.top();
                    if (curblock->blktype() == ASTBlock::BLK_EXCEPT)
                        curblock->setEnd(jump_target);
                }
            }
            break;
        case Pyc::LIST_APPEND:
        case Pyc::LIST_APPEND_A:
            {
                PycRef<ASTNode> value = stack.top();
                stack.pop();

                PycRef<ASTNode> list = (operand > 0) ? stack.top(operand) : stack.top();


                if (curblock->blktype() == ASTBlock::BLK_FOR
                        && curblock.cast<ASTIterBlock>()->isComprehension()) {
                    stack.pop();
                    stack.push(new ASTComprehension(value));
                } else {
                    PycRef<PycString> append_name = new PycString();
                    append_name->setValue("append");
                    ASTCall::pparam_t pparams;
                    pparams.push_back(value);
                    curblock->append(new ASTCall(
                        new ASTBinary(list, new ASTName(append_name), ASTBinary::BIN_ATTR),
                        pparams, ASTCall::kwparam_t()));
                }
            }
            break;
        case Pyc::SET_UPDATE_A:
            {
                PycRef<ASTNode> rhs = stack.top();
                stack.pop();
                PycRef<ASTSet> lhs = stack.top().cast<ASTSet>();
                stack.pop();

                if (rhs.type() != ASTNode::NODE_OBJECT) {
                    fprintf(stderr, "Unsupported argument found for SET_UPDATE\n");
                    break;
                }

                // I've only ever seen this be a TYPE_FROZENSET, but let's be careful...
                PycRef<PycObject> obj = rhs.cast<ASTObject>()->object();
                if (obj->type() != PycObject::TYPE_FROZENSET) {
                    fprintf(stderr, "Unsupported argument type found for SET_UPDATE\n");
                    break;
                }

                ASTSet::value_t result = lhs->values();
                for (const auto& it : obj.cast<PycSet>()->values()) {
                    result.push_back(new ASTObject(it));
                }

                stack.push(new ASTSet(result));
            }
            break;
        case Pyc::LIST_EXTEND_A:
            {
                PycRef<ASTNode> rhs = stack.top();
                stack.pop();
                PycRef<ASTList> lhs = stack.top().cast<ASTList>();
                stack.pop();

                if (rhs.type() != ASTNode::NODE_OBJECT) {
                    fprintf(stderr, "Unsupported argument found for LIST_EXTEND\n");
                    break;
                }

                // I've only ever seen this be a SMALL_TUPLE, but let's be careful...
                PycRef<PycObject> obj = rhs.cast<ASTObject>()->object();
                if (obj->type() != PycObject::TYPE_TUPLE && obj->type() != PycObject::TYPE_SMALL_TUPLE) {
                    fprintf(stderr, "Unsupported argument type found for LIST_EXTEND\n");
                    break;
                }

                ASTList::value_t result = lhs->values();
                for (const auto& it : obj.cast<PycTuple>()->values()) {
                    result.push_back(new ASTObject(it));
                }

                stack.push(new ASTList(result));
            }
            break;
        case Pyc::LOAD_ATTR_A:
            {
                PycRef<ASTNode> name = stack.top();
                if (name.type() != ASTNode::NODE_IMPORT) {
                    stack.pop();

                    if (mod->verCompare(3, 12) >= 0) {
                        if (operand & 1) {
                            /* Changed in version 3.12:
                            If the low bit of name is set, then a NULL or self is pushed to the stack
                            before the attribute or unbound method respectively. */
                            stack.push(nullptr);
                        }
                        operand >>= 1;
                    }

                    stack.push(new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR));
                }
            }
            break;
        case Pyc::LOAD_SUPER_ATTR_A:
        case Pyc::INSTRUMENTED_LOAD_SUPER_ATTR_A:
            {
                // CPython stack order (top->down): self, cls, global super
                PycRef<ASTNode> self_obj = stack.top();
                stack.pop();
                PycRef<ASTNode> cls_obj = stack.top();
                stack.pop();
                PycRef<ASTNode> global_super = stack.top();
                stack.pop();

                ASTCall::pparam_t super_args;
                if (operand & 2) {
                    // two-argument super(cls, self)
                    super_args.push_back(cls_obj);
                    super_args.push_back(self_obj);
                }
                PycRef<ASTNode> super_call = new ASTCall(global_super, super_args, ASTCall::kwparam_t());

                // For method-load form, keep CALL protocol with NULL sentinel.
                if (operand & 1) {
                    stack.push(nullptr);
                }

                int name_index = operand >> 2;
                stack.push(new ASTBinary(super_call, new ASTName(code->getName(name_index)),
                                         ASTBinary::BIN_ATTR));
            }
            break;
        case Pyc::LOAD_BUILD_CLASS:
            stack.push(new ASTLoadBuildClass(new PycObject()));
            break;
        case Pyc::LOAD_CLOSURE_A:
            /* Ignore this */
            break;
        case Pyc::LOAD_CONST_A:
            {
                PycRef<ASTObject> t_ob = new ASTObject(code->getConst(operand));

                if ((t_ob->object().type() == PycObject::TYPE_TUPLE ||
                        t_ob->object().type() == PycObject::TYPE_SMALL_TUPLE) &&
                        !t_ob->object().cast<PycTuple>()->values().size()) {
                    ASTTuple::value_t values;
                    stack.push(new ASTTuple(values));
                } else if (t_ob->object().type() == PycObject::TYPE_NONE) {
                    stack.push(NULL);
                } else {
                    stack.push(t_ob.cast<ASTNode>());
                }
            }
            break;
        case Pyc::LOAD_SMALL_INT_A:
            {
                PycRef<PycInt> intObj = new PycInt(operand);
                stack.push(new ASTObject(intObj.cast<PycObject>()));
            }
            break;
        case Pyc::LOAD_DEREF_A:
        case Pyc::LOAD_CLASSDEREF_A:
            stack.push(new ASTName(code->getCellVar(mod, operand)));
            break;
        case Pyc::LOAD_FAST_A:
        case Pyc::LOAD_FAST_CHECK_A:
            if (mod->verCompare(1, 3) < 0)
                stack.push(new ASTName(code->getName(operand)));
            else
                stack.push(new ASTName(code->getLocal(operand)));
            break;
        case Pyc::LOAD_FAST_AND_CLEAR_A:
            {
                PycRef<ASTNode> marker;
                if (mod->verCompare(1, 3) < 0)
                    marker = new ASTName(code->getName(operand));
                else
                    marker = new ASTName(code->getLocal(operand));
                cleared_fast_markers[operand] = marker;
                stack.push(marker);
            }
            break;
        case Pyc::LOAD_FAST_LOAD_FAST_A:
            stack.push(new ASTName(code->getLocal(operand >> 4)));
            stack.push(new ASTName(code->getLocal(operand & 0xF)));
            break;
        case Pyc::LOAD_GLOBAL_A:
            if (mod->verCompare(3, 11) >= 0) {
                // Loads the global named co_names[namei>>1] onto the stack.
                if (operand & 1) {
                    /* Changed in version 3.11: 
                    If the low bit of "NAMEI" (operand) is set, 
                    then a NULL is pushed to the stack before the global variable. */
                    stack.push(nullptr);
                }
                operand >>= 1;
            }
            stack.push(new ASTName(code->getName(operand)));
            break;
        case Pyc::LOAD_LOCALS:
            stack.push(new ASTNode(ASTNode::NODE_LOCALS));
            break;
        case Pyc::STORE_LOCALS:
            stack.pop();
            break;
        case Pyc::LOAD_METHOD_A:
            {
                // Behave like LOAD_ATTR
                PycRef<ASTNode> name = stack.top();
                stack.pop();
                stack.push(new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR));
            }
            break;
        case Pyc::LOAD_NAME_A:
            stack.push(new ASTName(code->getName(operand)));
            break;
        case Pyc::LOAD_COMMON_CONSTANT_A:
            {
                const char *common_name = NULL;
                switch (operand) {
                case 0:
                    common_name = "AssertionError";
                    break;
                case 1:
                    common_name = "NotImplementedError";
                    break;
                case 2:
                    common_name = "tuple";
                    break;
                case 3:
                    common_name = "all";
                    break;
                case 4:
                    common_name = "any";
                    break;
                case 5:
                    common_name = "list";
                    break;
                case 6:
                    common_name = "set";
                    break;
                default:
                    break;
                }

                PycRef<PycString> name = new PycString();
                if (common_name) {
                    name->setValue(common_name);
                } else {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "AUDIT_COMMON_CONSTANT_UNKNOWN_%d", operand);
                    name->setValue(buf);
                    fprintf(stderr, "AUDIT_WARN_LOAD_COMMON_CONSTANT_UNKNOWN operand=%d pos=%d curpos=%d\n",
                            operand, pos, curpos);
                }
                stack.push(new ASTName(name));
            }
            break;
        case Pyc::LOAD_ASSERTION_ERROR:
            {
                PycRef<PycString> assertion_name = new PycString();
                assertion_name->setValue("AssertionError");
                stack.push(new ASTName(assertion_name));
            }
            break;
        case Pyc::LOAD_SPECIAL_A:
            {
                /* LOAD_SPECIAL: load special method from TOS object.
                   Operand: 0=__enter__, 1=__exit__, 2=__aenter__, 3=__aexit__
                   Stack: owner -> self_or_null, bound_method  (net +1) */
                static const char* special_names[] = {
                    "__enter__", "__exit__", "__aenter__", "__aexit__"
                };
                const char* sname = (operand >= 0
                        && (size_t)operand < sizeof(special_names) / sizeof(special_names[0]))
                    ? special_names[operand] : "__special__";
                PycRef<PycString> attr_name = new PycString();
                attr_name->setValue(sname);
                PycRef<ASTNode> owner = stack.top();
                stack.pop();
                /* Push NULL for self_or_null so CALL_A pops it correctly */
                stack.push(nullptr);
                /* Push bound method as owner.<special> */
                stack.push(new ASTBinary(owner, new ASTName(attr_name),
                                         ASTBinary::BIN_ATTR));
            }
            break;
        case Pyc::MAKE_CLOSURE_A:
        case Pyc::MAKE_FUNCTION:
        case Pyc::MAKE_FUNCTION_A:
            {
                PycRef<ASTNode> fun_code = stack.top();
                stack.pop();

                /* Test for the qualified name of the function (at TOS) */
                int tos_type = fun_code.cast<ASTObject>()->object().type();
                if (tos_type != PycObject::TYPE_CODE &&
                    tos_type != PycObject::TYPE_CODE2) {
                    fun_code = stack.top();
                    stack.pop();
                }

                ASTFunction::defarg_t defArgs, kwDefArgs;
                const int defCount = operand & 0xFF;
                const int kwDefCount = (operand >> 8) & 0xFF;
                for (int i = 0; i < defCount; ++i) {
                    defArgs.push_front(stack.top());
                    stack.pop();
                }
                for (int i = 0; i < kwDefCount; ++i) {
                    kwDefArgs.push_front(stack.top());
                    stack.pop();
                }
                stack.push(new ASTFunction(fun_code, defArgs, kwDefArgs));
            }
            break;
        case Pyc::SET_FUNCTION_ATTRIBUTE_A:
            if (!stack.empty()) {
                PycRef<ASTNode> func = stack.top();
                stack.pop();

                PycRef<ASTNode> attr;
                if (!stack.empty()) {
                    attr = stack.top();
                    stack.pop();
                }

                if (func != NULL && func.type() == ASTNode::NODE_FUNCTION) {
                    PycRef<ASTFunction> fn = func.cast<ASTFunction>();
                    ASTFunction::defarg_t defArgs = fn->defargs();
                    ASTFunction::defarg_t kwDefArgs = fn->kwdefargs();

                    if ((operand & 0x01) && attr != NULL) {
                        defArgs.clear();
                        if (attr.type() == ASTNode::NODE_OBJECT) {
                            PycRef<PycObject> obj = attr.cast<ASTObject>()->object();
                            if (obj->type() == PycObject::TYPE_TUPLE
                                    || obj->type() == PycObject::TYPE_SMALL_TUPLE) {
                                std::vector<PycRef<PycObject>> values = obj.cast<PycTuple>()->values();
                                for (std::vector<PycRef<PycObject>>::const_iterator it = values.begin();
                                        it != values.end(); ++it) {
                                    defArgs.push_back(new ASTObject(*it));
                                }
                            }
                        } else if (attr.type() == ASTNode::NODE_TUPLE) {
                            ASTTuple::value_t values = attr.cast<ASTTuple>()->values();
                            for (ASTTuple::value_t::const_iterator it = values.begin();
                                    it != values.end(); ++it) {
                                defArgs.push_back(*it);
                            }
                        }
                    }

                    if ((operand & 0x02) && attr != NULL) {
                        kwDefArgs.clear();
                        std::unordered_map<std::string, PycRef<ASTNode>> defaultsByName;

                        if (attr.type() == ASTNode::NODE_OBJECT) {
                            PycRef<PycObject> obj = attr.cast<ASTObject>()->object();
                            if (obj->type() == PycObject::TYPE_DICT) {
                                PycDict::value_t values = obj.cast<PycDict>()->values();
                                for (PycDict::value_t::const_iterator it = values.begin();
                                        it != values.end(); ++it) {
                                    PycRef<PycObject> keyObj = std::get<0>(*it);
                                    PycRef<PycObject> valObj = std::get<1>(*it);
                                    if (keyObj->type() == PycObject::TYPE_STRING
                                            || keyObj->type() == PycObject::TYPE_ASCII
                                            || keyObj->type() == PycObject::TYPE_ASCII_INTERNED
                                            || keyObj->type() == PycObject::TYPE_SHORT_ASCII
                                            || keyObj->type() == PycObject::TYPE_SHORT_ASCII_INTERNED
                                            || keyObj->type() == PycObject::TYPE_UNICODE) {
                                        defaultsByName[keyObj.cast<PycString>()->value()] = new ASTObject(valObj);
                                    }
                                }
                            }
                        } else if (attr.type() == ASTNode::NODE_MAP) {
                            ASTMap::map_t values = attr.cast<ASTMap>()->values();
                            for (ASTMap::map_t::const_iterator it = values.begin();
                                    it != values.end(); ++it) {
                                std::string key;
                                if (it->first.type() == ASTNode::NODE_NAME) {
                                    key = it->first.cast<ASTName>()->name()->value();
                                } else if (it->first.type() == ASTNode::NODE_OBJECT) {
                                    PycRef<PycObject> keyObj = it->first.cast<ASTObject>()->object();
                                    if (keyObj->type() == PycObject::TYPE_STRING
                                            || keyObj->type() == PycObject::TYPE_ASCII
                                            || keyObj->type() == PycObject::TYPE_ASCII_INTERNED
                                            || keyObj->type() == PycObject::TYPE_SHORT_ASCII
                                            || keyObj->type() == PycObject::TYPE_SHORT_ASCII_INTERNED
                                            || keyObj->type() == PycObject::TYPE_UNICODE) {
                                        key = keyObj.cast<PycString>()->value();
                                    }
                                }
                                if (!key.empty())
                                    defaultsByName[key] = it->second;
                            }
                        }

                        PycRef<PycCode> code_src;
                        PycRef<ASTNode> fn_code = fn->code();
                        if (fn_code != NULL && fn_code.type() == ASTNode::NODE_OBJECT) {
                            PycRef<PycObject> fn_code_obj = fn_code.cast<ASTObject>()->object();
                            if (fn_code_obj->type() == PycObject::TYPE_CODE
                                    || fn_code_obj->type() == PycObject::TYPE_CODE2) {
                                code_src = fn_code_obj.cast<PycCode>();
                            }
                        }

                        if (code_src != NULL && code_src->kwOnlyArgCount() > 0) {
                            int first_kw = code_src->argCount();
                            for (int i = 0; i < code_src->kwOnlyArgCount(); ++i) {
                                std::string arg_name = code_src->getLocal(first_kw + i)->value();
                                std::unordered_map<std::string, PycRef<ASTNode>>::const_iterator it =
                                        defaultsByName.find(arg_name);
                                if (it != defaultsByName.end())
                                    kwDefArgs.push_back(it->second);
                            }
                        } else {
                            for (std::unordered_map<std::string, PycRef<ASTNode>>::const_iterator it =
                                    defaultsByName.begin(); it != defaultsByName.end(); ++it) {
                                kwDefArgs.push_back(it->second);
                            }
                        }
                    }

                    func = new ASTFunction(fn->code(), defArgs, kwDefArgs);
                }

                stack.push(func);
            }
            break;
        case Pyc::NOP:
            break;
        case Pyc::POP_BLOCK:
            {
                if (curblock->blktype() == ASTBlock::BLK_CONTAINER ||
                        curblock->blktype() == ASTBlock::BLK_FINALLY) {
                    /* These should only be popped by an END_FINALLY */
                    break;
                }

                if (curblock->blktype() == ASTBlock::BLK_WITH) {
                    // This should only be popped by a WITH_CLEANUP
                    break;
                }

                if (curblock->nodes().size() &&
                        curblock->nodes().back().type() == ASTNode::NODE_KEYWORD) {
                    curblock->removeLast();
                }

                if (curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELIF
                        || curblock->blktype() == ASTBlock::BLK_ELSE
                        || curblock->blktype() == ASTBlock::BLK_TRY
                        || curblock->blktype() == ASTBlock::BLK_EXCEPT
                        || curblock->blktype() == ASTBlock::BLK_FINALLY) {
                    if (!stack_hist.empty()) {
                        stack = stack_hist.top();
                        stack_hist.pop();
                    } else {
                        fprintf(stderr, "Warning: Stack history is empty, something wrong might have happened\n");
                    }
                }
                PycRef<ASTBlock> tmp = curblock;
                blocks.pop();

                if (!blocks.empty())
                    curblock = blocks.top();

                if (!(tmp->blktype() == ASTBlock::BLK_ELSE
                        && tmp->nodes().size() == 0)) {
                    curblock->append(tmp.cast<ASTNode>());
                }

                if (tmp->blktype() == ASTBlock::BLK_FOR && tmp->end() >= pos) {
                    stack_hist.push(stack);

                    PycRef<ASTBlock> blkelse = new ASTBlock(ASTBlock::BLK_ELSE, tmp->end());
                    blocks.push(blkelse);
                    curblock = blocks.top();
                }

                if (curblock->blktype() == ASTBlock::BLK_TRY
                        && tmp->blktype() != ASTBlock::BLK_FOR
                        && tmp->blktype() != ASTBlock::BLK_ASYNCFOR
                        && tmp->blktype() != ASTBlock::BLK_WHILE) {
                    stack = stack_hist.top();
                    stack_hist.pop();

                    tmp = curblock;
                    blocks.pop();
                    curblock = blocks.top();

                    if (!(tmp->blktype() == ASTBlock::BLK_ELSE
                            && tmp->nodes().size() == 0)) {
                        curblock->append(tmp.cast<ASTNode>());
                    }
                }

                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    PycRef<ASTContainerBlock> cont = curblock.cast<ASTContainerBlock>();

                    if (tmp->blktype() == ASTBlock::BLK_ELSE && !cont->hasFinally()) {

                        /* Pop the container */
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(cont.cast<ASTNode>());

                    } else if ((tmp->blktype() == ASTBlock::BLK_ELSE && cont->hasFinally())
                            || (tmp->blktype() == ASTBlock::BLK_TRY && !cont->hasExcept())) {

                        /* Add the finally block */
                        stack_hist.push(stack);

                        PycRef<ASTBlock> final = new ASTBlock(ASTBlock::BLK_FINALLY, 0, true);
                        blocks.push(final);
                        curblock = blocks.top();
                    }
                }

                if ((curblock->blktype() == ASTBlock::BLK_FOR || curblock->blktype() == ASTBlock::BLK_ASYNCFOR)
                        && curblock->end() == pos) {
                    blocks.pop();
                    blocks.top()->append(curblock.cast<ASTNode>());
                    curblock = blocks.top();
                }
            }
            break;
        case Pyc::POP_EXCEPT:
            /* Do nothing. */
            break;
        case Pyc::PUSH_EXC_INFO:
            if (mod->verCompare(3, 11) >= 0) {
                auto unwind_finished_conditionals = [&]() {
                    while (curblock->blktype() == ASTBlock::BLK_IF
                            || curblock->blktype() == ASTBlock::BLK_ELIF
                            || curblock->blktype() == ASTBlock::BLK_ELSE) {
                        if (curblock->end() == 0 || curblock->end() > curpos)
                            break;

                        PycRef<ASTBlock> prev_ctrl = curblock;
                        blocks.pop();
                        curblock = blocks.top();
                        curblock->append(prev_ctrl.cast<ASTNode>());
                        if (!stack_hist.empty())
                            stack_hist.pop();
                    }
                };

                const int incoming_handler_depth = exception_handler_depth_for_target(code, curpos);
                while (curblock->blktype() == ASTBlock::BLK_EXCEPT) {
                    PycRef<ASTCondBlock> except_block = curblock.cast<ASTCondBlock>();
                    if (incoming_handler_depth >= 0
                            && except_block->handlerDepth() >= 0
                            && except_block->handlerDepth() != incoming_handler_depth)
                        break;
                    const bool stale_placeholder = except_block->cond() == NULL
                            && curblock->end() != 0
                            && curblock->end() <= curpos;
                    const bool stale_typed_handler = except_block->cond() != NULL
                            && curblock->end() != 0
                            && curblock->end() <= curpos;
                    if (!stale_placeholder && !stale_typed_handler) {
                        break;
                    }

                    PycRef<ASTBlock> prev_except = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    if (stale_placeholder) {
                        PycRef<ASTNode> preserved_return = first_return_in_block(prev_except);
                        if (preserved_return != NULL)
                            curblock->append(preserved_return);
                    } else if (!stale_placeholder) {
                        curblock->append(prev_except.cast<ASTNode>());
                    }
                    if (!stack_hist.empty())
                        stack_hist.pop();
                    unwind_finished_conditionals();
                }
                unwind_finished_conditionals();
            }
            break;
        case Pyc::END_FOR:
            {
                stack.pop();

                if ((opcode == Pyc::END_FOR) && (mod->majorVer() == 3) && (mod->minorVer() == 12)) {
                    // one additional pop for python 3.12
                    stack.pop();
                }

                // end for loop here
                /* TODO : Ensure that FOR loop ends here. 
                   Due to CACHE instructions at play, the end indicated in
                   the for loop by pycdas is not correct, it is off by
                   some small amount. */
                if (curblock->blktype() == ASTBlock::BLK_FOR
                        || curblock->blktype() == ASTBlock::BLK_ASYNCFOR) {
                    PycRef<ASTBlock> prev = blocks.top();
                    blocks.pop();

                    curblock = blocks.top();
                    curblock->append(prev.cast<ASTNode>());
                }
                else if (mod->verCompare(3, 11) < 0) {
                    fprintf(stderr, "Wrong block type %i for END_FOR\n", curblock->blktype());
                }
            }
            break;
        case Pyc::POP_TOP:
            {
                PycRef<ASTNode> value = stack.top();
                stack.pop();

                if (!curblock->inited()) {
                    if (curblock->blktype() == ASTBlock::BLK_WITH) {
                        curblock.cast<ASTWithBlock>()->setExpr(value);
                    } else {
                        curblock->init();
                    }
                    break;
                } else if (value == nullptr || value->processed()) {
                    break;
                }

                curblock->append(value);

                if (curblock->blktype() == ASTBlock::BLK_FOR
                        && curblock.cast<ASTIterBlock>()->isComprehension()) {
                    /* This relies on some really uncertain logic...
                     * If it's a comprehension, the only POP_TOP should be
                     * a call to append the iter to the list.
                     */
                    if (value.type() == ASTNode::NODE_CALL) {
                        auto& pparams = value.cast<ASTCall>()->pparams();
                        if (!pparams.empty()) {
                            PycRef<ASTNode> res = pparams.front();
                            stack.push(new ASTComprehension(res));
                        }
                    }
                }
            }
            break;
        case Pyc::PRINT_ITEM:
            {
                PycRef<ASTPrint> printNode;
                if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                    printNode = curblock->nodes().back().try_cast<ASTPrint>();
                if (printNode && printNode->stream() == nullptr && !printNode->eol())
                    printNode->add(stack.top());
                else
                    curblock->append(new ASTPrint(stack.top()));
                stack.pop();
            }
            break;
        case Pyc::PRINT_ITEM_TO:
            {
                PycRef<ASTNode> stream = stack.top();
                stack.pop();

                PycRef<ASTPrint> printNode;
                if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                    printNode = curblock->nodes().back().try_cast<ASTPrint>();
                if (printNode && printNode->stream() == stream && !printNode->eol())
                    printNode->add(stack.top());
                else
                    curblock->append(new ASTPrint(stack.top(), stream));
                stack.pop();
                if (stream)
                    stream->setProcessed();
            }
            break;
        case Pyc::PRINT_NEWLINE:
            {
                PycRef<ASTPrint> printNode;
                if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                    printNode = curblock->nodes().back().try_cast<ASTPrint>();
                if (printNode && printNode->stream() == nullptr && !printNode->eol())
                    printNode->setEol(true);
                else
                    curblock->append(new ASTPrint(nullptr));
                stack.pop();
            }
            break;
        case Pyc::PRINT_NEWLINE_TO:
            {
                PycRef<ASTNode> stream = stack.top();
                stack.pop();

                PycRef<ASTPrint> printNode;
                if (curblock->size() > 0 && curblock->nodes().back().type() == ASTNode::NODE_PRINT)
                    printNode = curblock->nodes().back().try_cast<ASTPrint>();
                if (printNode && printNode->stream() == stream && !printNode->eol())
                    printNode->setEol(true);
                else
                    curblock->append(new ASTPrint(nullptr, stream));
                stack.pop();
                if (stream)
                    stream->setProcessed();
            }
            break;
        case Pyc::RAISE_VARARGS_A:
            {
                ASTRaise::param_t paramList;
                for (int i = 0; i < operand; i++) {
                    paramList.push_front(stack.top());
                    stack.pop();
                }
                curblock->append(new ASTRaise(paramList));

                if ((curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && stack_hist.size()
                        && (mod->verCompare(2, 6) >= 0)) {
                    stack = stack_hist.top();
                    stack_hist.pop();

                    PycRef<ASTBlock> prev = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(prev.cast<ASTNode>());
                }
            }
            break;
        case Pyc::RERAISE:
        case Pyc::RERAISE_A:
            {
                curblock->append(new ASTRaise(ASTRaise::param_t()));

                if ((curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && stack_hist.size()
                        && (mod->verCompare(2, 6) >= 0)) {
                    stack = stack_hist.top();
                    stack_hist.pop();

                    PycRef<ASTBlock> prev = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(prev.cast<ASTNode>());
                }
            }
            break;
        case Pyc::RETURN_VALUE:
        case Pyc::INSTRUMENTED_RETURN_VALUE_A:
            {
                PycRef<ASTNode> value = stack.top();
                stack.pop();
                curblock->append(new ASTReturn(value));

                if ((curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && stack_hist.size()
                        && (mod->verCompare(2, 6) >= 0)) {
                    stack = stack_hist.top();
                    stack_hist.pop();

                    PycRef<ASTBlock> prev = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(prev.cast<ASTNode>());
                }
            }
            break;
        case Pyc::RETURN_CONST_A:
        case Pyc::INSTRUMENTED_RETURN_CONST_A:
            {
                PycRef<ASTObject> value = new ASTObject(code->getConst(operand));
                curblock->append(new ASTReturn(value.cast<ASTNode>()));

                if ((curblock->blktype() == ASTBlock::BLK_IF
                        || curblock->blktype() == ASTBlock::BLK_ELSE)
                        && stack_hist.size()
                        && (mod->verCompare(2, 6) >= 0)) {
                    stack = stack_hist.top();
                    stack_hist.pop();

                    PycRef<ASTBlock> prev = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(prev.cast<ASTNode>());
                }
            }
            break;
        case Pyc::ROT_TWO:
            {
                PycRef<ASTNode> one = stack.top();
                stack.pop();
                if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    stack.pop();
                }
                PycRef<ASTNode> two = stack.top();
                stack.pop();

                stack.push(one);
                stack.push(two);
            }
            break;
        case Pyc::ROT_THREE:
            {
                PycRef<ASTNode> one = stack.top();
                stack.pop();
                PycRef<ASTNode> two = stack.top();
                stack.pop();
                if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    stack.pop();
                }
                PycRef<ASTNode> three = stack.top();
                stack.pop();
                stack.push(one);
                stack.push(three);
                stack.push(two);
            }
            break;
        case Pyc::ROT_FOUR:
            {
                PycRef<ASTNode> one = stack.top();
                stack.pop();
                PycRef<ASTNode> two = stack.top();
                stack.pop();
                PycRef<ASTNode> three = stack.top();
                stack.pop();
                if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                    stack.pop();
                }
                PycRef<ASTNode> four = stack.top();
                stack.pop();
                stack.push(one);
                stack.push(four);
                stack.push(three);
                stack.push(two);
            }
            break;
        case Pyc::SET_LINENO_A:
            // Ignore
            break;
        case Pyc::SETUP_WITH_A:
            {
                PycRef<ASTBlock> withblock = new ASTWithBlock(pos+operand);
                blocks.push(withblock);
                curblock = blocks.top();
            }
            break;
        case Pyc::BEFORE_WITH:
            {
                /* 3.11+ stack effect:
                   mgr -> exit_func, enter_result
                   We reconstruct this explicitly to avoid hard-failing on BEFORE_WITH. */
                PycRef<ASTNode> mgr = stack.top();
                stack.pop();

                PycRef<PycString> enter_name = new PycString();
                enter_name->setValue("__enter__");
                PycRef<PycString> exit_name = new PycString();
                exit_name->setValue("__exit__");

                ASTCall::pparam_t no_params;
                stack.push(new ASTBinary(mgr, new ASTName(exit_name), ASTBinary::BIN_ATTR));
                stack.push(new ASTCall(
                    new ASTBinary(mgr, new ASTName(enter_name), ASTBinary::BIN_ATTR),
                    no_params, ASTCall::kwparam_t()));
            }
            break;
        case Pyc::WITH_EXCEPT_START:
            /* Python 3.11+ with-cleanup handler internals are skipped. */
            skip_with_except_cleanup = true;
            break;
        case Pyc::WITH_CLEANUP:
        case Pyc::WITH_CLEANUP_START:
            {
                // Stack top should be a None. Ignore it.
                PycRef<ASTNode> none = stack.top();
                stack.pop();

                if (none != NULL) {
                    fprintf(stderr, "AUDIT_WARN_WITH_CLEANUP_NONE_NOT_NULL pos=%d curpos=%d opcode=%s\n",
                            pos, curpos, Pyc::OpcodeName(opcode));
                    cleanBuild = false;
                    break;
                }

                if (curblock->blktype() == ASTBlock::BLK_WITH
                        && curblock->end() == curpos) {
                    PycRef<ASTBlock> with = curblock;
                    blocks.pop();
                    curblock = blocks.top();
                    curblock->append(with.cast<ASTNode>());
                }
                else {
                    fprintf(stderr, "AUDIT_WARN_WITH_CLEANUP_NO_MATCH pos=%d curpos=%d opcode=%s\n",
                            pos, curpos, Pyc::OpcodeName(opcode));
                    cleanBuild = false;
                }
            }
            break;
        case Pyc::WITH_CLEANUP_FINISH:
            /* Ignore this */
            break;
        case Pyc::SETUP_EXCEPT_A:
            {
                if (curblock->blktype() == ASTBlock::BLK_CONTAINER) {
                    curblock.cast<ASTContainerBlock>()->setExcept(pos+operand);
                } else {
                    PycRef<ASTBlock> next = new ASTContainerBlock(0, pos+operand);
                    blocks.push(next.cast<ASTBlock>());
                }

                /* Store the current stack for the except/finally statement(s) */
                stack_hist.push(stack);
                PycRef<ASTBlock> tryblock = new ASTBlock(ASTBlock::BLK_TRY, pos+operand, true);
                blocks.push(tryblock.cast<ASTBlock>());
                curblock = blocks.top();

                need_try = false;
            }
            break;
        case Pyc::SETUP_FINALLY_A:
            {
                PycRef<ASTBlock> next = new ASTContainerBlock(pos+operand);
                blocks.push(next.cast<ASTBlock>());
                curblock = blocks.top();

                need_try = true;
            }
            break;
        case Pyc::SETUP_LOOP_A:
            {
                PycRef<ASTBlock> next = new ASTCondBlock(ASTBlock::BLK_WHILE, pos+operand, NULL, false);
                blocks.push(next.cast<ASTBlock>());
                curblock = blocks.top();
            }
            break;
        case Pyc::SLICE_0:
            {
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE0);
                stack.push(new ASTSubscr(name, slice));
            }
            break;
        case Pyc::SLICE_1:
            {
                PycRef<ASTNode> lower = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE1, lower);
                stack.push(new ASTSubscr(name, slice));
            }
            break;
        case Pyc::SLICE_2:
            {
                PycRef<ASTNode> upper = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE2, NULL, upper);
                stack.push(new ASTSubscr(name, slice));
            }
            break;
        case Pyc::SLICE_3:
            {
                PycRef<ASTNode> upper = stack.top();
                stack.pop();
                PycRef<ASTNode> lower = stack.top();
                stack.pop();
                PycRef<ASTNode> name = stack.top();
                stack.pop();

                PycRef<ASTNode> slice = new ASTSlice(ASTSlice::SLICE3, lower, upper);
                stack.push(new ASTSubscr(name, slice));
            }
            break;
        case Pyc::STORE_ATTR_A:
            {
                if (unpack) {
                    PycRef<ASTNode> name = stack.top();
                    stack.pop();
                    PycRef<ASTNode> attr = new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR);

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(attr);
                    else {
                        fprintf(stderr, "AUDIT_WARN_UNPACK_TARGET_NOT_TUPLE pos=%d curpos=%d opcode=%s\n",
                                pos, curpos, Pyc::OpcodeName(opcode));
                        cleanBuild = false;
                    }

                    if (--unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();
                        if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> name = stack.top();
                    stack.pop();
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    PycRef<ASTNode> attr = new ASTBinary(name, new ASTName(code->getName(operand)), ASTBinary::BIN_ATTR);
                    if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, attr, stack, curblock);
                    } else {
                        curblock->append(new ASTStore(value, attr));
                    }
                }
            }
            break;
        case Pyc::STORE_DEREF_A:
            {
                if (unpack) {
                    PycRef<ASTNode> name = new ASTName(code->getCellVar(mod, operand));

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() != ASTNode::NODE_TUPLE) {
                        // Some optimized STORE_* variants may bypass tuple placeholder creation.
                        // Recreate it here so unpack assignment can still be reconstructed.
                        stack.pop();
                        ASTTuple::value_t vals;
                        tup = new ASTTuple(vals);
                        stack.push(tup);
                    }
                    tup.cast<ASTTuple>()->add(name);

                    if (--unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    PycRef<ASTNode> name = new ASTName(code->getCellVar(mod, operand));

                    if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else {
                        curblock->append(new ASTStore(value, name));
                    }
                }
            }
            break;
        case Pyc::STORE_FAST_A:
            {
                if (unpack) {
                    PycRef<ASTNode> name;

                    if (mod->verCompare(1, 3) < 0)
                        name = new ASTName(code->getName(operand));
                    else
                        name = new ASTName(code->getLocal(operand));

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() != ASTNode::NODE_TUPLE) {
                        // Optimized STORE_* sequences may skip placeholder creation.
                        // Recreate tuple target so unpack assignment can continue.
                        stack.pop();
                        ASTTuple::value_t vals;
                        tup = new ASTTuple(vals);
                        stack.push(tup);
                    }
                    tup.cast<ASTTuple>()->add(name);

                    if (--unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    PycRef<ASTNode> name;

                    if (mod->verCompare(1, 3) < 0)
                        name = new ASTName(code->getName(operand));
                    else
                        name = new ASTName(code->getLocal(operand));

                    int marker_index = -1;
                    PycRef<ASTNode> marker_value;
                    for (const auto& it : cleared_fast_markers) {
                        if (value == it.second) {
                            marker_index = it.first;
                            marker_value = it.second;
                            break;
                        }
                    }
                    if (marker_index < 0 && value != NULL && value.type() == ASTNode::NODE_NAME) {
                        for (const auto& it : cleared_fast_markers) {
                            if (it.second != NULL && it.second.type() == ASTNode::NODE_NAME
                                    && value.cast<ASTName>()->name()->isEqual(
                                        it.second.cast<ASTName>()->name()->value())) {
                                marker_index = it.first;
                                marker_value = it.second;
                                break;
                            }
                        }
                    }

                    if (marker_index >= 0) {
                        // Restore store for LOAD_FAST_AND_CLEAR temp: suppress local restoration.
                        if (marker_index == operand) {
                            cleared_fast_markers.erase(marker_index);
                            break;
                        }

                        // Pattern: STORE_FAST <real_target>; STORE_FAST <cleared_local>
                        // Rebind first store to the actual value under the restoration marker.
                        if (!stack.empty()) {
                            PycBuffer lookahead = source;
                            int next_opcode = -1, next_operand = 0, look_pos = pos;
                            do {
                                bc_next(lookahead, mod, next_opcode, next_operand, look_pos);
                            } while (!lookahead.atEof() && next_opcode == Pyc::CACHE);

                            if (next_opcode == Pyc::STORE_FAST_A && next_operand == marker_index) {
                                PycRef<ASTNode> reassigned_value = stack.top();
                                stack.pop();
                                value = reassigned_value;
                                stack.push(marker_value);
                            }
                        }
                    }

                    if (name.cast<ASTName>()->name()->value()[0] == '_'
                            && name.cast<ASTName>()->name()->value()[1] == '[') {
                        /* Don't show stores of list comp append objects. */
                        break;
                    }

                    if (curblock->blktype() == ASTBlock::BLK_FOR
                            && !curblock->inited()) {
                        curblock.cast<ASTIterBlock>()->setIndex(name);
                    } else if (curblock->blktype() == ASTBlock::BLK_WITH
                                   && !curblock->inited()) {
                        curblock.cast<ASTWithBlock>()->setExpr(value);
                        curblock.cast<ASTWithBlock>()->setVar(name);
                    } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else {
                        curblock->append(new ASTStore(value, name));
                    }
                }
            }
            break;
        case Pyc::STORE_FAST_LOAD_FAST_A:
            {
                if (unpack > 0) {
                    PycRef<ASTNode> name = new ASTName(code->getLocal(operand >> 4));
                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() != ASTNode::NODE_TUPLE) {
                        stack.pop();
                        ASTTuple::value_t vals;
                        tup = new ASTTuple(vals);
                        stack.push(tup);
                    }
                    tup.cast<ASTTuple>()->add(name);

                    if (--unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }

                    stack.push(new ASTName(code->getLocal(operand & 0xF)));
                    break;
                }

                PycRef<ASTNode> value = stack.top();
                stack.pop();

                PycRef<ASTNode> name = new ASTName(code->getLocal(operand >> 4));
                if (curblock->blktype() == ASTBlock::BLK_FOR
                        && !curblock->inited()) {
                    curblock.cast<ASTIterBlock>()->setIndex(name);
                } else if (curblock->blktype() == ASTBlock::BLK_WITH
                               && !curblock->inited()) {
                    curblock.cast<ASTWithBlock>()->setExpr(value);
                    curblock.cast<ASTWithBlock>()->setVar(name);
                } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                    append_to_chain_store(value, name, stack, curblock);
                } else {
                    curblock->append(new ASTStore(value, name));
                }

                stack.push(new ASTName(code->getLocal(operand & 0xF)));
            }
            break;
        case Pyc::STORE_FAST_STORE_FAST_A:
            {
                if (unpack > 0) {
                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() != ASTNode::NODE_TUPLE) {
                        stack.pop();
                        ASTTuple::value_t vals;
                        tup = new ASTTuple(vals);
                        stack.push(tup);
                    }
                    tup.cast<ASTTuple>()->add(new ASTName(code->getLocal(operand >> 4)));
                    tup.cast<ASTTuple>()->add(new ASTName(code->getLocal(operand & 0xF)));

                    unpack -= 2;
                    if (unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                    break;
                }

                PycRef<ASTNode> value1 = stack.top();
                stack.pop();
                PycRef<ASTNode> value2 = stack.top();
                stack.pop();

                PycRef<ASTNode> name1 = new ASTName(code->getLocal(operand >> 4));
                PycRef<ASTNode> name2 = new ASTName(code->getLocal(operand & 0xF));

                if (value1.type() == ASTNode::NODE_CHAINSTORE) {
                    append_to_chain_store(value1, name1, stack, curblock);
                } else {
                    curblock->append(new ASTStore(value1, name1));
                }
                if (value2.type() == ASTNode::NODE_CHAINSTORE) {
                    append_to_chain_store(value2, name2, stack, curblock);
                } else {
                    curblock->append(new ASTStore(value2, name2));
                }
            }
            break;
        case Pyc::STORE_GLOBAL_A:
            {
                PycRef<ASTNode> name = new ASTName(code->getName(operand));

                if (unpack) {
                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(name);
                    else {
                        fprintf(stderr, "AUDIT_WARN_UNPACK_TARGET_NOT_TUPLE pos=%d curpos=%d opcode=%s\n",
                                pos, curpos, Pyc::OpcodeName(opcode));
                        cleanBuild = false;
                    }

                    if (--unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else {
                        curblock->append(new ASTStore(value, name));
                    }
                }

                /* Mark the global as used */
                code->markGlobal(name.cast<ASTName>()->name());
            }
            break;
        case Pyc::STORE_NAME_A:
            {
                if (unpack) {
                    PycRef<ASTNode> name = new ASTName(code->getName(operand));

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(name);
                    else {
                        fprintf(stderr, "AUDIT_WARN_UNPACK_TARGET_NOT_TUPLE pos=%d curpos=%d opcode=%s\n",
                                pos, curpos, Pyc::OpcodeName(opcode));
                        cleanBuild = false;
                    }

                    if (--unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();

                        if (curblock->blktype() == ASTBlock::BLK_FOR
                                && !curblock->inited()) {
                            PycRef<ASTTuple> tuple = tup.try_cast<ASTTuple>();
                            if (tuple != NULL)
                                tuple->setRequireParens(false);
                            curblock.cast<ASTIterBlock>()->setIndex(tup);
                        } else if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();

                    PycRef<PycString> varname = code->getName(operand);
                    if (varname->length() >= 2 && varname->value()[0] == '_'
                            && varname->value()[1] == '[') {
                        /* Don't show stores of list comp append objects. */
                        break;
                    }

                    // Return private names back to their original name
                    const std::string class_prefix = std::string("_") + code->name()->strValue();
                    if (varname->startsWith(class_prefix + std::string("__")))
                        varname->setValue(varname->strValue().substr(class_prefix.size()));

                    PycRef<ASTNode> name = new ASTName(varname);

                    if (curblock->blktype() == ASTBlock::BLK_FOR
                            && !curblock->inited()) {
                        curblock.cast<ASTIterBlock>()->setIndex(name);
                    } else if (stack.top().type() == ASTNode::NODE_IMPORT) {
                        PycRef<ASTImport> import = stack.top().cast<ASTImport>();

                        import->add_store(new ASTStore(value, name));
                    } else if (curblock->blktype() == ASTBlock::BLK_WITH
                               && !curblock->inited()) {
                        curblock.cast<ASTWithBlock>()->setExpr(value);
                        curblock.cast<ASTWithBlock>()->setVar(name);
                    } else if (value.type() == ASTNode::NODE_CHAINSTORE) {
                        append_to_chain_store(value, name, stack, curblock);
                    } else {
                        curblock->append(new ASTStore(value, name));

                        if (value.type() == ASTNode::NODE_INVALID)
                            break;
                    }
                }
            }
            break;
        case Pyc::STORE_SLICE_0:
            {
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                PycRef<ASTNode> value = stack.top();
                stack.pop();

                curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE0))));
            }
            break;
        case Pyc::STORE_SLICE_1:
            {
                PycRef<ASTNode> upper = stack.top();
                stack.pop();
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                PycRef<ASTNode> value = stack.top();
                stack.pop();

                curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE1, upper))));
            }
            break;
        case Pyc::STORE_SLICE_2:
            {
                PycRef<ASTNode> lower = stack.top();
                stack.pop();
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                PycRef<ASTNode> value = stack.top();
                stack.pop();

                curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE2, NULL, lower))));
            }
            break;
        case Pyc::STORE_SLICE_3:
            {
                PycRef<ASTNode> lower = stack.top();
                stack.pop();
                PycRef<ASTNode> upper = stack.top();
                stack.pop();
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                PycRef<ASTNode> value = stack.top();
                stack.pop();

                curblock->append(new ASTStore(value, new ASTSubscr(dest, new ASTSlice(ASTSlice::SLICE3, upper, lower))));
            }
            break;
        case Pyc::STORE_SUBSCR:
            {
                if (unpack) {
                    PycRef<ASTNode> subscr = stack.top();
                    stack.pop();
                    PycRef<ASTNode> dest = stack.top();
                    stack.pop();

                    PycRef<ASTNode> save = new ASTSubscr(dest, subscr);

                    PycRef<ASTNode> tup = stack.top();
                    if (tup.type() == ASTNode::NODE_TUPLE)
                        tup.cast<ASTTuple>()->add(save);
                    else {
                        fprintf(stderr, "AUDIT_WARN_UNPACK_TARGET_NOT_TUPLE pos=%d curpos=%d opcode=%s\n",
                                pos, curpos, Pyc::OpcodeName(opcode));
                        cleanBuild = false;
                    }

                    if (--unpack <= 0) {
                        stack.pop();
                        PycRef<ASTNode> seq = stack.top();
                        stack.pop();
                        if (seq.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(seq, tup, stack, curblock);
                        } else {
                            curblock->append(new ASTStore(seq, tup));
                        }
                    }
                } else {
                    PycRef<ASTNode> subscr = stack.top();
                    stack.pop();
                    PycRef<ASTNode> dest = stack.top();
                    stack.pop();
                    PycRef<ASTNode> src = stack.top();
                    stack.pop();

                    // If variable annotations are enabled, we'll need to check for them here.
                    // Python handles a varaible annotation by setting:
                    // __annotations__['var-name'] = type
                    const bool found_annotated_var = (variable_annotations && dest->type() == ASTNode::Type::NODE_NAME
                                                      && dest.cast<ASTName>()->name()->isEqual("__annotations__"));

                    if (found_annotated_var) {
                        // Annotations can be done alone or as part of an assignment.
                        // In the case of an assignment, we'll see a NODE_STORE on the stack.
                        if (!curblock->nodes().empty() && curblock->nodes().back()->type() == ASTNode::Type::NODE_STORE) {
                            // Replace the existing NODE_STORE with a new one that includes the annotation.
                            PycRef<ASTStore> store = curblock->nodes().back().cast<ASTStore>();
                            curblock->removeLast();
                            curblock->append(new ASTStore(store->src(),
                                                          new ASTAnnotatedVar(subscr, src)));
                        } else {
                            curblock->append(new ASTAnnotatedVar(subscr, src));
                        }
                    } else {
                        if (dest.type() == ASTNode::NODE_MAP) {
                            dest.cast<ASTMap>()->add(subscr, src);
                        } else if (src.type() == ASTNode::NODE_CHAINSTORE) {
                            append_to_chain_store(src, new ASTSubscr(dest, subscr), stack, curblock);
                        } else {
                            curblock->append(new ASTStore(src, new ASTSubscr(dest, subscr)));
                        }
                    }
                }
            }
            break;
        case Pyc::TO_BOOL:
            break;
        case Pyc::UNARY_CALL:
            {
                PycRef<ASTNode> func = stack.top();
                stack.pop();
                stack.push(new ASTCall(func, ASTCall::pparam_t(), ASTCall::kwparam_t()));
            }
            break;
        case Pyc::UNARY_CONVERT:
            {
                PycRef<ASTNode> name = stack.top();
                stack.pop();
                stack.push(new ASTConvert(name));
            }
            break;
        case Pyc::UNARY_INVERT:
            {
                PycRef<ASTNode> arg = stack.top();
                stack.pop();
                stack.push(new ASTUnary(arg, ASTUnary::UN_INVERT));
            }
            break;
        case Pyc::UNARY_NEGATIVE:
            {
                PycRef<ASTNode> arg = stack.top();
                stack.pop();
                stack.push(new ASTUnary(arg, ASTUnary::UN_NEGATIVE));
            }
            break;
        case Pyc::UNARY_NOT:
            {
                PycRef<ASTNode> arg = stack.top();
                stack.pop();
                stack.push(new ASTUnary(arg, ASTUnary::UN_NOT));
            }
            break;
        case Pyc::UNARY_POSITIVE:
            {
                PycRef<ASTNode> arg = stack.top();
                stack.pop();
                stack.push(new ASTUnary(arg, ASTUnary::UN_POSITIVE));
            }
            break;
        case Pyc::UNPACK_LIST_A:
        case Pyc::UNPACK_TUPLE_A:
        case Pyc::UNPACK_SEQUENCE_A:
        case Pyc::UNPACK_EX_A:
            {
                if (opcode == Pyc::UNPACK_EX_A) {
                    unpack = (operand & 0xFF) + ((operand >> 8) & 0xFF) + 1;
                } else {
                    unpack = operand;
                }
                if (unpack > 0) {
                    ASTTuple::value_t vals;
                    stack.push(new ASTTuple(vals));
                } else {
                    // Unpack zero values and assign it to top of stack or for loop variable.
                    // E.g. [] = TOS / for [] in X
                    ASTTuple::value_t vals;
                    auto tup = new ASTTuple(vals);
                    if (curblock->blktype() == ASTBlock::BLK_FOR
                        && !curblock->inited()) {
                        tup->setRequireParens(true);
                        curblock.cast<ASTIterBlock>()->setIndex(tup);
                    } else if (stack.top().type() == ASTNode::NODE_CHAINSTORE) {
                        auto chainStore = stack.top();
                        stack.pop();
                        append_to_chain_store(chainStore, tup, stack, curblock);
                    } else {
                        curblock->append(new ASTStore(stack.top(), tup));
                        stack.pop();
                    }
                }
            }
            break;
        case Pyc::YIELD_FROM:
            {
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                // TODO: Support yielding into a non-null destination
                PycRef<ASTNode> value = stack.top();
                if (value) {
                    value->setProcessed();
                    curblock->append(new ASTReturn(value, ASTReturn::YIELD_FROM));
                }
            }
            break;
        case Pyc::YIELD_VALUE:
            {
                PycRef<ASTNode> value = stack.top();
                stack.pop();
                curblock->append(new ASTReturn(value, ASTReturn::YIELD));
            }
            break;
        case Pyc::YIELD_VALUE_A:
        case Pyc::INSTRUMENTED_YIELD_VALUE_A:
            {
                if (!stack.empty()) {
                    PycRef<ASTNode> value = stack.top();
                    stack.pop();
                    curblock->append(new ASTReturn(value, ASTReturn::YIELD));
                }
            }
            break;
        case Pyc::SETUP_ANNOTATIONS:
            variable_annotations = true;
            break;
        case Pyc::PRECALL_A:
        case Pyc::RESUME_A:
        case Pyc::INSTRUMENTED_RESUME_A:
        case Pyc::COPY_FREE_VARS_A:
        case Pyc::MAKE_CELL_A:
        case Pyc::RETURN_GENERATOR:
            /* We just entirely ignore this / no-op */
            break;
        case Pyc::CACHE:
            /* These "fake" opcodes are used as placeholders for optimizing
               certain opcodes in Python 3.11+.  Since we have no need for
               that during disassembly/decompilation, we can just treat these
               as no-ops. */
            break;
        case Pyc::PUSH_NULL:
            stack.push(nullptr);
            break;
        case Pyc::GEN_START_A:
            stack.pop();
            break;
        case Pyc::SWAP_A:
            {
                if (operand <= 1)
                    break;
                if (unpack > 0)
                    break;

                auto is_store_opcode = [](int op) -> bool {
                    switch (op) {
                    case Pyc::STORE_ATTR_A:
                    case Pyc::STORE_DEREF_A:
                    case Pyc::STORE_FAST_A:
                    case Pyc::STORE_FAST_LOAD_FAST_A:
                    case Pyc::STORE_FAST_STORE_FAST_A:
                    case Pyc::STORE_GLOBAL_A:
                    case Pyc::STORE_NAME_A:
                    case Pyc::STORE_SLICE:
                    case Pyc::STORE_SUBSCR:
                        return true;
                    default:
                        return false;
                    }
                };

                bool contains_null = false;
                for (int i = 1; i <= operand; ++i) {
                    if (stack.top(i) == NULL) {
                        contains_null = true;
                        break;
                    }
                }

                int next_opcode = -1;
                int next_operand = 0;
                if (!source.atEof()) {
                    PycBuffer lookahead = source;
                    int lookpos = pos;
                    bc_next(lookahead, mod, next_opcode, next_operand, lookpos);
                }

                if (mod->verCompare(3, 11) >= 0 && next_opcode == Pyc::POP_EXCEPT)
                    break;

                if (!contains_null && (next_opcode == Pyc::SWAP_A || is_store_opcode(next_opcode))) {
                    unpack = operand;
                    ASTTuple::value_t values;
                    values.resize(operand);
                    for (int i = 0; i < operand; i++) {
                        values[operand - i - 1] = stack.top();
                        stack.pop();
                    }
                    auto tup = new ASTTuple(values);
                    tup->setRequireParens(false);
                    auto targets = new ASTTuple(ASTTuple::value_t());
                    targets->setRequireParens(false);
                    stack.push(tup);
                    stack.push(targets);
                    break;
                }

                PycRef<ASTNode> tos = stack.top();
                stack.pop();

                ASTTuple::value_t middle;
                for (int i = 0; i < operand - 2; i++) {
                    middle.push_back(stack.top());
                    stack.pop();
                }

                PycRef<ASTNode> nth = stack.top();
                stack.pop();

                stack.push(tos);
                for (ASTTuple::value_t::reverse_iterator it = middle.rbegin();
                        it != middle.rend(); ++it) {
                    stack.push(*it);
                }
                stack.push(nth);
            }
            break;
        case Pyc::BINARY_SLICE:
            {
                PycRef<ASTNode> end = stack.top();
                stack.pop();
                PycRef<ASTNode> start = stack.top();
                stack.pop();
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                PycRef<ASTNode> slice = build_slice_node(start, end);
                stack.push(new ASTSubscr(dest, slice));
            }
            break;
        case Pyc::STORE_SLICE:
            {
                PycRef<ASTNode> end = stack.top();
                stack.pop();
                PycRef<ASTNode> start = stack.top();
                stack.pop();
                PycRef<ASTNode> dest = stack.top();
                stack.pop();
                PycRef<ASTNode> values = stack.top();
                stack.pop();
                PycRef<ASTNode> slice = build_slice_node(start, end);
                curblock->append(new ASTStore(values, new ASTSubscr(dest, slice)));
            }
            break;
        case Pyc::COPY_A:
            {
                PycRef<ASTNode> value = stack.top(operand);
                stack.push(value);
            }
            break;
        case Pyc::CALL_INTRINSIC_1_A:
            {
                static const char *s_intrinsic1[] = {
                    "INTRINSIC_1_INVALID", "INTRINSIC_PRINT", "INTRINSIC_IMPORT_STAR",
                    "INTRINSIC_STOPITERATION_ERROR", "INTRINSIC_ASYNC_GEN_WRAP",
                    "INTRINSIC_UNARY_POSITIVE", "INTRINSIC_LIST_TO_TUPLE", "INTRINSIC_TYPEVAR",
                    "INTRINSIC_PARAMSPEC", "INTRINSIC_TYPEVARTUPLE",
                    "INTRINSIC_SUBSCRIPT_GENERIC", "INTRINSIC_TYPEALIAS",
                };
                static const size_t s_intrinsic1_len = sizeof(s_intrinsic1) / sizeof(s_intrinsic1[0]);

                const char *fname = (static_cast<size_t>(operand) < s_intrinsic1_len)
                                    ? s_intrinsic1[operand] : nullptr;
                PycRef<PycString> name = new PycString();
                if (fname)
                    name->setValue(fname);
                else {
                    char buf[40];
                    snprintf(buf, sizeof(buf), "pycdc_intrinsic1_unknown");
                    name->setValue(buf);
                }

                PycRef<ASTNode> arg = stack.top();
                stack.pop();
                ASTCall::pparam_t pparams;
                pparams.push_back(arg);
                stack.push(new ASTCall(new ASTName(name), pparams, ASTCall::kwparam_t()));
            }
            break;
        case Pyc::CALL_INTRINSIC_2_A:
            {
                static const char *s_intrinsic2[] = {
                    "INTRINSIC_2_INVALID", "INTRINSIC_PREP_RERAISE_STAR",
                    "INTRINSIC_TYPEVAR_WITH_BOUND", "INTRINSIC_TYPEVAR_WITH_CONSTRAINTS",
                    "INTRINSIC_SET_FUNCTION_TYPE_PARAMS", "INTRINSIC_SET_TYPEPARAM_DEFAULT",
                };
                static const size_t s_intrinsic2_len = sizeof(s_intrinsic2) / sizeof(s_intrinsic2[0]);

                const char *fname = (static_cast<size_t>(operand) < s_intrinsic2_len)
                                    ? s_intrinsic2[operand] : nullptr;
                PycRef<PycString> name = new PycString();
                if (fname)
                    name->setValue(fname);
                else {
                    char buf[40];
                    snprintf(buf, sizeof(buf), "pycdc_intrinsic2_unknown");
                    name->setValue(buf);
                }

                // TOS is second arg, TOS1 is first arg (preserve argument order)
                PycRef<ASTNode> arg2 = stack.top();
                stack.pop();
                PycRef<ASTNode> arg1 = stack.top();
                stack.pop();
                ASTCall::pparam_t pparams;
                pparams.push_back(arg1);
                pparams.push_back(arg2);
                stack.push(new ASTCall(new ASTName(name), pparams, ASTCall::kwparam_t()));
            }
            break;
        default:
            fprintf(stderr, "Unsupported opcode: %s (%d)\n", Pyc::OpcodeName(opcode), opcode);
            cleanBuild = false;
            return new ASTNodeList(defblock->nodes());
        }

        else_pop =  ( (curblock->blktype() == ASTBlock::BLK_ELSE)
                      || (curblock->blktype() == ASTBlock::BLK_IF)
                      || (curblock->blktype() == ASTBlock::BLK_ELIF) )
                 && (curblock->end() == pos);
    }

    if (skip_with_except_cleanup) {
        fprintf(stderr, "AUDIT_WARN_WITH_EXCEPT_CLEANUP_UNTERMINATED pos=%d\n", pos);
        cleanBuild = false;
    }

    if (stack_hist.size()) {
        if (mod->verCompare(3, 11) < 0) {
            fprintf(stderr, "AUDIT_WARN_STACK_HISTORY_NON_EMPTY count=%llu\n", stack_hist.size());
            cleanBuild = false;
        }

        while (stack_hist.size()) {
            stack_hist.pop();
        }
    }

    g_ast_append_offset_hint = -1;

    if (blocks.size() > 1) {
        if (mod->verCompare(3, 11) < 0) {
            fprintf(stderr, "AUDIT_WARN_BLOCK_STACK_NON_EMPTY count=%llu\n", blocks.size() - 1);
            cleanBuild = false;
        }

        while (blocks.size() > 1) {
            PycRef<ASTBlock> tmp = blocks.top();
            blocks.pop();

            blocks.top()->append(tmp.cast<ASTNode>());
        }
    }

    ASTBlock::list_t outNodes(defblock->nodes());

    if (mod->verCompare(3, 11) >= 0) {
        bool hasTryBlock = false;
        ASTBlock::list_t::iterator firstExcept = outNodes.end();

        for (ASTBlock::list_t::iterator it = outNodes.begin(); it != outNodes.end(); ++it) {
            PycRef<ASTBlock> blk = (*it).try_cast<ASTBlock>();
            if (blk == NULL)
                continue;
            if (blk->blktype() == ASTBlock::BLK_TRY)
                hasTryBlock = true;
            if (blk->blktype() == ASTBlock::BLK_EXCEPT) {
                firstExcept = it;
                break;
            }
        }

        if (!hasTryBlock && firstExcept != outNodes.begin() && firstExcept != outNodes.end()) {
            ASTBlock::list_t prefix_nodes;
            ASTBlock::list_t try_nodes;
            int protected_start = first_depth_zero_protected_start(code);
            for (ASTBlock::list_t::iterator it = outNodes.begin(); it != firstExcept; ) {
                ASTBlock::list_t::iterator cur = it++;
                int off = node_first_offset(*cur);
                if (protected_start >= 0 && off >= 0 && off < protected_start)
                    prefix_nodes.emplace_back(*cur);
                else
                    try_nodes.emplace_back(*cur);
                outNodes.erase(cur);
            }

            if (try_nodes.empty()) {
                try_nodes.swap(prefix_nodes);
                prefix_nodes.clear();
            }

            PycRef<ASTBlock> syntheticTry = new ASTBlock(ASTBlock::BLK_TRY, 0, true);
            syntheticTry->init();
            for (const auto& node : try_nodes)
                syntheticTry->append(node);

            ASTBlock::list_t::iterator insert_it = outNodes.begin();
            for (const auto& node : prefix_nodes)
                outNodes.insert(insert_it, node);
            outNodes.insert(insert_it, syntheticTry.cast<ASTNode>());
        }
    }

#if defined(BLOCK_DEBUG) || defined(STACK_DEBUG)
    fprintf(stderr, "=== End BuildFromCode %s ===\n", (code != NULL && code->name() != NULL)
            ? code->name()->value() : "<unnamed>");
#endif

    return new ASTNodeList(outNodes);
}

static void append_to_chain_store(const PycRef<ASTNode> &chainStore,
        PycRef<ASTNode> item, FastStack& stack, const PycRef<ASTBlock>& curblock)
{
    stack.pop();    // ignore identical source object.
    chainStore.cast<ASTChainStore>()->append(item);
    if (stack.top().type() == PycObject::TYPE_NULL) {
        curblock->append(chainStore);
    } else {
        stack.push(chainStore);
    }
}

static int cmp_prec(PycRef<ASTNode> parent, PycRef<ASTNode> child)
{
    /* Determine whether the parent has higher precedence than therefore
       child, so we don't flood the source code with extraneous parens.
       Else we'd have expressions like (((a + b) + c) + d) when therefore
       equivalent, a + b + c + d would suffice. */

    if (parent.type() == ASTNode::NODE_UNARY && parent.cast<ASTUnary>()->op() == ASTUnary::UN_NOT)
        return 1;   // Always parenthesize not(x)
    if (child.type() == ASTNode::NODE_BINARY) {
        PycRef<ASTBinary> binChild = child.cast<ASTBinary>();
        if (parent.type() == ASTNode::NODE_BINARY) {
            PycRef<ASTBinary> binParent = parent.cast<ASTBinary>();
            if (binParent->right() == child) {
                if (binParent->op() == ASTBinary::BIN_SUBTRACT &&
                    binChild->op() == ASTBinary::BIN_ADD)
                    return 1;
                else if (binParent->op() == ASTBinary::BIN_DIVIDE &&
                         binChild->op() == ASTBinary::BIN_MULTIPLY)
                    return 1;
            }
            return binChild->op() - binParent->op();
        }
        else if (parent.type() == ASTNode::NODE_COMPARE)
            return (binChild->op() == ASTBinary::BIN_LOG_AND ||
                    binChild->op() == ASTBinary::BIN_LOG_OR) ? 1 : -1;
        else if (parent.type() == ASTNode::NODE_UNARY)
            return (binChild->op() == ASTBinary::BIN_POWER) ? -1 : 1;
    } else if (child.type() == ASTNode::NODE_UNARY) {
        PycRef<ASTUnary> unChild = child.cast<ASTUnary>();
        if (parent.type() == ASTNode::NODE_BINARY) {
            PycRef<ASTBinary> binParent = parent.cast<ASTBinary>();
            if (binParent->op() == ASTBinary::BIN_LOG_AND ||
                binParent->op() == ASTBinary::BIN_LOG_OR)
                return -1;
            else if (unChild->op() == ASTUnary::UN_NOT)
                return 1;
            else if (binParent->op() == ASTBinary::BIN_POWER)
                return 1;
            else
                return -1;
        } else if (parent.type() == ASTNode::NODE_COMPARE) {
            return (unChild->op() == ASTUnary::UN_NOT) ? 1 : -1;
        } else if (parent.type() == ASTNode::NODE_UNARY) {
            return unChild->op() - parent.cast<ASTUnary>()->op();
        }
    } else if (child.type() == ASTNode::NODE_COMPARE) {
        PycRef<ASTCompare> cmpChild = child.cast<ASTCompare>();
        if (parent.type() == ASTNode::NODE_BINARY)
            return (parent.cast<ASTBinary>()->op() == ASTBinary::BIN_LOG_AND ||
                    parent.cast<ASTBinary>()->op() == ASTBinary::BIN_LOG_OR) ? -1 : 1;
        else if (parent.type() == ASTNode::NODE_COMPARE)
            return cmpChild->op() - parent.cast<ASTCompare>()->op();
        else if (parent.type() == ASTNode::NODE_UNARY)
            return (parent.cast<ASTUnary>()->op() == ASTUnary::UN_NOT) ? -1 : 1;
    }

    /* For normal nodes, don't parenthesize anything */
    return -1;
}

static void print_ordered(PycRef<ASTNode> parent, PycRef<ASTNode> child,
                          PycModule* mod, std::ostream& pyc_output)
{
    if (child.type() == ASTNode::NODE_BINARY ||
        child.type() == ASTNode::NODE_COMPARE) {
        if (cmp_prec(parent, child) > 0) {
            pyc_output << "(";
            print_src(child, mod, pyc_output);
            pyc_output << ")";
        } else {
            print_src(child, mod, pyc_output);
        }
    } else if (child.type() == ASTNode::NODE_UNARY) {
        if (cmp_prec(parent, child) > 0) {
            pyc_output << "(";
            print_src(child, mod, pyc_output);
            pyc_output << ")";
        } else {
            print_src(child, mod, pyc_output);
        }
    } else {
        print_src(child, mod, pyc_output);
    }
}

static void start_line(int indent, std::ostream& pyc_output)
{
    if (inLambda)
        return;
    for (int i=0; i<indent; i++)
        pyc_output << "    ";
}

static void end_line(std::ostream& pyc_output)
{
    if (inLambda)
        return;
    pyc_output << "\n";
}

int cur_indent = -1;
static int loop_depth = 0;

static void collect_direct_store_names(const PycRef<ASTBlock>& blk,
        std::unordered_set<std::string>& out_names)
{
    if (blk == NULL)
        return;
    for (const auto& ln : blk->nodes()) {
        PycRef<ASTStore> st = ln.try_cast<ASTStore>();
        if (st == NULL || st->dest() == NULL || st->dest().type() != ASTNode::NODE_NAME)
            continue;
        out_names.insert(st->dest().cast<ASTName>()->name()->value());
    }
}

static bool store_dest_contains_name(const PycRef<ASTNode>& dest,
        const std::unordered_set<std::string>& names)
{
    if (dest == NULL)
        return false;
    if (dest.type() == ASTNode::NODE_NAME)
        return names.find(dest.cast<ASTName>()->name()->value()) != names.end();
    if (dest.type() == ASTNode::NODE_TUPLE) {
        for (const auto& elt : dest.cast<ASTTuple>()->values()) {
            if (store_dest_contains_name(elt, names))
                return true;
        }
    }
    return false;
}

static bool node_uses_any_name(const PycRef<ASTNode>& node,
        const std::unordered_set<std::string>& names)
{
    for (const auto& name : names) {
        if (node_uses_name(node, name.c_str()))
            return true;
    }
    return false;
}

static void normalize_except_rejoin_tail(ASTBlock::list_t& lines)
{
    for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTBlock> tryblk = (*it).try_cast<ASTBlock>();
        if (tryblk == NULL || tryblk->blktype() != ASTBlock::BLK_TRY)
            continue;

        ASTBlock::list_t::iterator ex_begin = it;
        ++ex_begin;
        ASTBlock::list_t::iterator ex_end = ex_begin;
        size_t except_count = 0;
        std::unordered_set<std::string> shared_names;
        bool initialized_shared = false;
        bool all_non_terminal = true;

        while (ex_end != lines.end()) {
            PycRef<ASTBlock> exblk = (*ex_end).try_cast<ASTBlock>();
            if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
                break;
            ++except_count;
            if (block_has_terminal_stmt(exblk)) {
                all_non_terminal = false;
                break;
            }
            std::unordered_set<std::string> block_names;
            collect_direct_store_names(exblk, block_names);
            if (block_names.empty()) {
                all_non_terminal = false;
                break;
            }
            if (!initialized_shared) {
                shared_names.swap(block_names);
                initialized_shared = true;
            } else {
                std::unordered_set<std::string> intersection;
                for (const auto& name : shared_names) {
                    if (block_names.find(name) != block_names.end())
                        intersection.insert(name);
                }
                shared_names.swap(intersection);
            }
            ++ex_end;
        }

        if (except_count < 2 || !all_non_terminal || shared_names.empty())
            continue;

        size_t total_nodes = 0;
        size_t last_store_idx = (size_t)-1;
        size_t idx = 0;
        for (const auto& ln : tryblk->nodes()) {
            ++total_nodes;
            PycRef<ASTStore> st = ln.try_cast<ASTStore>();
            if (st != NULL && st->dest() != NULL && st->dest().type() == ASTNode::NODE_NAME) {
                if (shared_names.find(st->dest().cast<ASTName>()->name()->value()) != shared_names.end())
                    last_store_idx = idx;
            }
            ++idx;
        }
        if (last_store_idx == (size_t)-1 || last_store_idx + 1 >= total_nodes)
            continue;

        ASTBlock::list_t prefix;
        ASTBlock::list_t tail;
        idx = 0;
        for (const auto& ln : tryblk->nodes()) {
            if (idx <= last_store_idx)
                prefix.emplace_back(ln);
            else
                tail.emplace_back(ln);
            ++idx;
        }
        if (tail.empty())
            continue;

        bool tail_uses_shared = false;
        for (const auto& ln : tail) {
            if (node_uses_any_name(ln, shared_names)) {
                tail_uses_shared = true;
                break;
            }
        }
        if (!tail_uses_shared)
            continue;

        while (tryblk->size() > 0)
            tryblk->removeLast();
        for (const auto& ln : prefix)
            tryblk->append(ln);

        for (ASTBlock::list_t::iterator jt = ex_begin; jt != ex_end; ++jt) {
            PycRef<ASTBlock> exblk = (*jt).cast<ASTBlock>();
            while (exblk->size() > 0 && is_pass_only_node(exblk->nodes().back()))
                exblk->removeLast();
            if (block_has_terminal_stmt(exblk))
                continue;
            for (const auto& ln : tail)
                exblk->append(ln);
        }
    }
}

static ASTBlock::list_t normalize_except_sequences(ASTBlock::list_t lines, PycModule* mod)
{
    (void)mod;
    normalize_except_rejoin_tail(lines);
    normalize_except_return_rejoin(lines);
    ASTBlock::list_t reshaped;
    for (ASTBlock::list_t::const_iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTBlock> condblk = (*it).try_cast<ASTBlock>();
        if (condblk != NULL
                && (condblk->blktype() == ASTBlock::BLK_IF
                    || condblk->blktype() == ASTBlock::BLK_ELIF)) {
            ASTBlock::list_t::const_iterator next = it;
            ++next;
            if (next != lines.end()) {
                PycRef<ASTBlock> elseblk = (*next).try_cast<ASTBlock>();
                if (elseblk != NULL
                        && elseblk->blktype() == ASTBlock::BLK_ELSE
                        && condblk->size() > 0) {
                    ASTBlock::list_t except_nodes;
                    ASTBlock::list_t tail_nodes;
                    bool collecting_except = true;
                    for (const auto& child : elseblk->nodes()) {
                        PycRef<ASTBlock> child_blk = child.try_cast<ASTBlock>();
                        if (collecting_except && child_blk != NULL && child_blk->blktype() == ASTBlock::BLK_EXCEPT) {
                            except_nodes.emplace_back(child);
                            continue;
                        }
                        collecting_except = false;
                        if (!is_pass_only_node(child))
                            tail_nodes.emplace_back(child);
                    }

                    if (!except_nodes.empty()) {
                        PycRef<ASTBlock> synthetic_try = new ASTBlock(ASTBlock::BLK_TRY);
                        synthetic_try->init();
                        synthetic_try->append(*it);
                        reshaped.emplace_back(synthetic_try.cast<ASTNode>());
                        for (const auto& ex : except_nodes)
                            reshaped.emplace_back(ex);
                        for (const auto& tail : tail_nodes)
                            reshaped.emplace_back(tail);
                        it = next;
                        continue;
                    }
                }
            }
        }
        reshaped.emplace_back(*it);
    }

    lines = std::move(reshaped);
    ASTBlock::list_t normalized;
    for (ASTBlock::list_t::const_iterator it = lines.begin(); it != lines.end();) {
        PycRef<ASTBlock> blk = (*it).try_cast<ASTBlock>();
        if (blk == NULL || blk->blktype() != ASTBlock::BLK_EXCEPT) {
            normalized.emplace_back(*it);
            ++it;
            continue;
        }

        ASTBlock::list_t except_chain;
        ASTBlock::list_t::const_iterator jt = it;
        while (jt != lines.end()) {
            PycRef<ASTBlock> exblk = (*jt).try_cast<ASTBlock>();
            if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
                break;
            ASTBlock::list_t::const_iterator next = jt;
            ++next;
            const bool has_later_except = (next != lines.end()
                    && (*next).try_cast<ASTBlock>() != NULL
                    && (*next).try_cast<ASTBlock>()->blktype() == ASTBlock::BLK_EXCEPT);
            if (!(has_later_except && is_bare_reraise_except_block(exblk)))
                except_chain.emplace_back(*jt);
            ++jt;
        }

        bool has_try_anchor = false;
        if (!normalized.empty()) {
            PycRef<ASTBlock> prev_blk = normalized.back().try_cast<ASTBlock>();
            has_try_anchor = (prev_blk != NULL && prev_blk->blktype() == ASTBlock::BLK_TRY);
        }

        if (!has_try_anchor && !normalized.empty()) {
            int protected_start = first_depth_zero_protected_start(cleanup_current_code);
            ASTBlock::list_t::iterator split = normalized.end();
            ASTBlock::list_t::iterator scan = normalized.end();
            bool has_non_cleanup_stmt = false;
            while (scan != normalized.begin()) {
                ASTBlock::list_t::iterator prev = scan;
                --prev;
                int prev_off = node_first_offset(*prev);
                if (protected_start >= 0 && prev_off >= 0 && prev_off < protected_start)
                    break;
                PycRef<ASTBlock> prev_blk = (*prev).try_cast<ASTBlock>();
                if (prev_blk != NULL) {
                    if (prev_blk->blktype() == ASTBlock::BLK_EXCEPT
                            || prev_blk->blktype() == ASTBlock::BLK_FINALLY
                            || prev_blk->blktype() == ASTBlock::BLK_CONTAINER) {
                        break;
                    }
                    if (prev_blk->blktype() == ASTBlock::BLK_TRY) {
                        ASTBlock::list_t::iterator next = prev;
                        ++next;
                        if (next != normalized.end()) {
                            PycRef<ASTBlock> next_blk = (*next).try_cast<ASTBlock>();
                            if (next_blk != NULL && next_blk->blktype() == ASTBlock::BLK_EXCEPT)
                                break;
                        }
                    }
                }
                if (*prev != NULL
                        && (*prev).type() != ASTNode::NODE_RAISE
                        && ((*prev).type() != ASTNode::NODE_KEYWORD
                            || ((*prev).cast<ASTKeyword>()->key() != ASTKeyword::KW_PASS
                                && (*prev).cast<ASTKeyword>()->key() != ASTKeyword::KW_CONTINUE
                                && (*prev).cast<ASTKeyword>()->key() != ASTKeyword::KW_BREAK))) {
                    has_non_cleanup_stmt = true;
                }
                split = prev;
                scan = prev;
            }

            if (split != normalized.end() && has_non_cleanup_stmt) {
                PycRef<ASTBlock> synthetic_try = new ASTBlock(ASTBlock::BLK_TRY, 0, true);
                synthetic_try->init();
                for (ASTBlock::list_t::iterator mv = split; mv != normalized.end(); ++mv)
                    synthetic_try->append(*mv);
                normalized.erase(split, normalized.end());
                normalized.emplace_back(synthetic_try.cast<ASTNode>());
            }
        }

        for (const auto& ex : except_chain)
            normalized.emplace_back(ex);
        it = jt;
    }
    return normalized;
}

static bool is_terminal_stmt(const PycRef<ASTNode>& node)
{
    if (node == NULL)
        return false;
    if (node.type() == ASTNode::NODE_RETURN || node.type() == ASTNode::NODE_RAISE)
        return true;
    if (node.type() == ASTNode::NODE_KEYWORD) {
        ASTKeyword::Word kw = node.cast<ASTKeyword>()->key();
        return kw == ASTKeyword::KW_BREAK || kw == ASTKeyword::KW_CONTINUE;
    }
    return false;
}

static bool is_none_like_node(const PycRef<ASTNode>& node)
{
    if (node == NULL)
        return true;
    if (node.type() == ASTNode::NODE_OBJECT) {
        PycRef<ASTObject> obj = node.cast<ASTObject>();
        return obj->object() == Pyc_None || obj->object().type() == PycObject::TYPE_NONE;
    }
    if (node.type() == ASTNode::NODE_NAME)
        return node.cast<ASTName>()->name()->isEqual("None");
    return false;
}

static bool is_all_none_like_node(const PycRef<ASTNode>& node)
{
    if (is_none_like_node(node))
        return true;
    if (node == NULL)
        return false;
    if (node.type() == ASTNode::NODE_TUPLE) {
        const ASTTuple::value_t& vals = node.cast<ASTTuple>()->values();
        if (vals.empty())
            return false;
        for (const auto& val : vals) {
            if (!is_none_like_node(val))
                return false;
        }
        return true;
    }
    if (node.type() == ASTNode::NODE_OBJECT) {
        PycRef<PycObject> obj = node.cast<ASTObject>()->object();
        if (obj != NULL
                && (obj->type() == PycObject::TYPE_TUPLE
                    || obj->type() == PycObject::TYPE_SMALL_TUPLE)) {
            PycRef<PycTuple> tup = obj.cast<PycTuple>();
            if (tup == NULL || tup->size() == 0)
                return false;
            for (int i = 0; i < tup->size(); ++i) {
                PycRef<PycObject> elem = tup->get(i);
                if (elem != Pyc_None && (elem == NULL || elem->type() != PycObject::TYPE_NONE))
                    return false;
            }
            return true;
        }
    }
    return false;
}

static bool extract_none_init_name(const PycRef<ASTNode>& node, PycRef<PycString>& out_name)
{
    if (node == NULL || node.type() != ASTNode::NODE_STORE)
        return false;
    PycRef<ASTStore> st = node.cast<ASTStore>();
    if (st->dest() == NULL || st->dest().type() != ASTNode::NODE_NAME)
        return false;
    if (!is_none_like_node(st->src()))
        return false;
    out_name = st->dest().cast<ASTName>()->name();
    return true;
}

static bool is_delete_of_name(const PycRef<ASTNode>& node, const PycRef<PycString>& name)
{
    if (node == NULL || name == NULL || node.type() != ASTNode::NODE_DELETE)
        return false;
    PycRef<ASTDelete> del = node.cast<ASTDelete>();
    return del->value() != NULL
            && del->value().type() == ASTNode::NODE_NAME
            && del->value().cast<ASTName>()->name()->isEqual(name->value());
}

static void strip_trailing_none_cleanup(ASTBlock::list_t& lines)
{
    while (!lines.empty()) {
        bool removed = false;
        if (lines.size() >= 2) {
            ASTBlock::list_t::iterator del_it = lines.end();
            --del_it;
            ASTBlock::list_t::iterator store_it = del_it;
            --store_it;
            PycRef<PycString> cleanup_name;
            if (extract_none_init_name(*store_it, cleanup_name)
                    && is_delete_of_name(*del_it, cleanup_name)) {
                lines.erase(del_it);
                lines.erase(store_it);
                removed = true;
            }
        }
        if (removed)
            continue;

        ASTBlock::list_t::iterator tail_it = lines.end();
        --tail_it;
        PycRef<PycString> cleanup_name;
        if (extract_none_init_name(*tail_it, cleanup_name)) {
            lines.erase(tail_it);
            continue;
        }
        break;
    }
}

static bool is_synthetic_loop_keyword_node(const PycRef<ASTNode>& node);

static void strip_trailing_synthetic_loop_keywords(ASTBlock::list_t& lines)
{
    while (!lines.empty()) {
        ASTBlock::list_t::iterator tail_it = lines.end();
        --tail_it;
        if (!is_synthetic_loop_keyword_node(*tail_it))
            break;
        lines.erase(tail_it);
    }
}

static bool node_uses_name(const PycRef<ASTNode>& node, const char* name)
{
    if (node == NULL || name == NULL)
        return false;

    switch (node.type()) {
    case ASTNode::NODE_NAME:
        return node.cast<ASTName>()->name()->isEqual(name);
    case ASTNode::NODE_STORE:
        return node_uses_name(node.cast<ASTStore>()->src(), name)
                || node_uses_name(node.cast<ASTStore>()->dest(), name);
    case ASTNode::NODE_DELETE:
        return node_uses_name(node.cast<ASTDelete>()->value(), name);
    case ASTNode::NODE_RETURN:
        return node_uses_name(node.cast<ASTReturn>()->value(), name);
    case ASTNode::NODE_UNARY:
        return node_uses_name(node.cast<ASTUnary>()->operand(), name);
    case ASTNode::NODE_BINARY:
    case ASTNode::NODE_COMPARE:
        return node_uses_name(node.cast<ASTBinary>()->left(), name)
                || node_uses_name(node.cast<ASTBinary>()->right(), name);
    case ASTNode::NODE_CALL:
        {
            PycRef<ASTCall> call = node.cast<ASTCall>();
            if (node_uses_name(call->func(), name))
                return true;
            for (const auto& param : call->pparams()) {
                if (node_uses_name(param, name))
                    return true;
            }
            for (const auto& kw : call->kwparams()) {
                if (node_uses_name(kw.first, name) || node_uses_name(kw.second, name))
                    return true;
            }
            return node_uses_name(call->var(), name) || node_uses_name(call->kw(), name);
        }
    case ASTNode::NODE_TUPLE:
        for (const auto& item : node.cast<ASTTuple>()->values()) {
            if (node_uses_name(item, name))
                return true;
        }
        return false;
    case ASTNode::NODE_LIST:
        for (const auto& item : node.cast<ASTList>()->values()) {
            if (node_uses_name(item, name))
                return true;
        }
        return false;
    case ASTNode::NODE_SET:
        for (const auto& item : node.cast<ASTSet>()->values()) {
            if (node_uses_name(item, name))
                return true;
        }
        return false;
    case ASTNode::NODE_MAP:
        for (const auto& kv : node.cast<ASTMap>()->values()) {
            if (node_uses_name(kv.first, name) || node_uses_name(kv.second, name))
                return true;
        }
        return false;
    case ASTNode::NODE_CONST_MAP:
        {
            PycRef<ASTConstMap> cmap = node.cast<ASTConstMap>();
            if (node_uses_name(cmap->keys(), name))
                return true;
            for (const auto& v : cmap->values()) {
                if (node_uses_name(v, name))
                    return true;
            }
            return false;
        }
    case ASTNode::NODE_SUBSCR:
        return node_uses_name(node.cast<ASTSubscr>()->name(), name)
                || node_uses_name(node.cast<ASTSubscr>()->key(), name);
    case ASTNode::NODE_NODELIST:
        for (const auto& ln : node.cast<ASTNodeList>()->nodes()) {
            if (node_uses_name(ln, name))
                return true;
        }
        return false;
    case ASTNode::NODE_BLOCK:
        for (const auto& ln : node.cast<ASTBlock>()->nodes()) {
            if (node_uses_name(ln, name))
                return true;
        }
        return false;
    case ASTNode::NODE_AWAITABLE:
        return node_uses_name(node.cast<ASTAwaitable>()->expression(), name);
    case ASTNode::NODE_FORMATTEDVALUE:
        return node_uses_name(node.cast<ASTFormattedValue>()->val(), name)
                || node_uses_name(node.cast<ASTFormattedValue>()->format_spec(), name);
    case ASTNode::NODE_JOINEDSTR:
        for (const auto& val : node.cast<ASTJoinedStr>()->values()) {
            if (node_uses_name(val, name))
                return true;
        }
        return false;
    default:
        return false;
    }
}

static std::string node_signature(const PycRef<ASTNode>& node)
{
    if (node == NULL)
        return "null";

    switch (node.type()) {
    case ASTNode::NODE_NAME:
        return std::string("N:") + node.cast<ASTName>()->name()->value();
    case ASTNode::NODE_OBJECT:
        {
            PycRef<PycObject> obj = node.cast<ASTObject>()->object();
            if (obj == Pyc_None || obj->type() == PycObject::TYPE_NONE)
                return "O:None";
            if (obj->type() == PycObject::TYPE_STRING
                    || obj->type() == PycObject::TYPE_UNICODE
                    || obj->type() == PycObject::TYPE_INTERNED
                    || obj->type() == PycObject::TYPE_ASCII
                    || obj->type() == PycObject::TYPE_ASCII_INTERNED
                    || obj->type() == PycObject::TYPE_SHORT_ASCII
                    || obj->type() == PycObject::TYPE_SHORT_ASCII_INTERNED)
                return std::string("S:") + obj.cast<PycString>()->value();
            return std::string("O:") + std::to_string(obj->type());
        }
    case ASTNode::NODE_BINARY:
    case ASTNode::NODE_COMPARE:
        {
            PycRef<ASTBinary> bin = node.cast<ASTBinary>();
            return std::string("B:")
                    + std::to_string(bin->op()) + "("
                    + node_signature(bin->left()) + ","
                    + node_signature(bin->right()) + ")";
        }
    case ASTNode::NODE_CALL:
        {
            PycRef<ASTCall> call = node.cast<ASTCall>();
            std::string sig = "C:" + node_signature(call->func()) + "(";
            bool first = true;
            for (const auto& param : call->pparams()) {
                if (!first)
                    sig += ",";
                sig += node_signature(param);
                first = false;
            }
            for (const auto& kw : call->kwparams()) {
                if (!first)
                    sig += ",";
                sig += node_signature(kw.first) + "=" + node_signature(kw.second);
                first = false;
            }
            if (call->hasVar()) {
                if (!first)
                    sig += ",";
                sig += "*" + node_signature(call->var());
                first = false;
            }
            if (call->hasKW()) {
                if (!first)
                    sig += ",";
                sig += "**" + node_signature(call->kw());
            }
            sig += ")";
            return sig;
        }
    case ASTNode::NODE_STORE:
        {
            PycRef<ASTStore> st = node.cast<ASTStore>();
            return std::string("ST:")
                    + node_signature(st->dest()) + "="
                    + node_signature(st->src());
        }
    case ASTNode::NODE_RETURN:
        {
            PycRef<ASTReturn> ret = node.cast<ASTReturn>();
            return std::string("R:")
                    + std::to_string(ret->rettype()) + "("
                    + node_signature(ret->value()) + ")";
        }
    case ASTNode::NODE_KEYWORD:
        return std::string("K:") + std::to_string(node.cast<ASTKeyword>()->key());
    case ASTNode::NODE_TUPLE:
        {
            std::string sig = "TU(";
            bool first = true;
            for (const auto& item : node.cast<ASTTuple>()->values()) {
                if (!first)
                    sig += ",";
                sig += node_signature(item);
                first = false;
            }
            sig += ")";
            return sig;
        }
    case ASTNode::NODE_SUBSCR:
        return std::string("SUB:")
                + node_signature(node.cast<ASTSubscr>()->name()) + "["
                + node_signature(node.cast<ASTSubscr>()->key()) + "]";
    default:
        return std::string("T:") + std::to_string(node.type());
    }
}

static bool is_pass_keyword_node(const PycRef<ASTNode>& node)
{
    return (node != NULL && node.type() == ASTNode::NODE_KEYWORD
            && node.cast<ASTKeyword>()->key() == ASTKeyword::KW_PASS);
}

static bool is_pass_only_node(const PycRef<ASTNode>& node)
{
    if (node == NULL)
        return false;
    if (is_pass_keyword_node(node))
        return true;
    if (node.type() == ASTNode::NODE_NODELIST) {
        const ASTNodeList::list_t& nodes = node.cast<ASTNodeList>()->nodes();
        if (nodes.empty())
            return false;
        for (const auto& child : nodes) {
            if (!is_pass_only_node(child))
                return false;
        }
        return true;
    }
    if (node.type() == ASTNode::NODE_BLOCK) {
        PycRef<ASTBlock> blk = node.cast<ASTBlock>();
        if (blk->size() == 0)
            return false;
        for (const auto& child : blk->nodes()) {
            if (!is_pass_only_node(child))
                return false;
        }
        return true;
    }
    return false;
}

static bool is_synthetic_loop_keyword_node(const PycRef<ASTNode>& node)
{
    if (node == NULL || node.type() != ASTNode::NODE_KEYWORD)
        return false;
    PycRef<ASTKeyword> kw = node.cast<ASTKeyword>();
    return kw->key() == ASTKeyword::KW_CONTINUE || kw->key() == ASTKeyword::KW_BREAK;
}

static void trim_synthetic_except_tail(const PycRef<ASTBlock>& blk)
{
    while (blk != NULL && blk->size() > 0) {
        const PycRef<ASTNode>& tail = blk->nodes().back();
        if (!is_pass_only_node(tail)
                && !is_synthetic_loop_keyword_node(tail))
            break;
        blk->removeLast();
    }
}

static bool block_has_terminal_stmt_before_synthetic_except_tail(const PycRef<ASTBlock>& blk)
{
    if (blk == NULL || blk->nodes().empty())
        return false;
    for (ASTBlock::list_t::const_reverse_iterator it = blk->nodes().rbegin();
            it != blk->nodes().rend(); ++it) {
        if (*it == NULL)
            continue;
        if (is_pass_only_node(*it) || is_synthetic_loop_keyword_node(*it))
            continue;
        return is_terminal_stmt(*it);
    }
    return false;
}

static bool is_bare_reraise_except_block(const PycRef<ASTBlock>& blk)
{
    if (blk == NULL || blk->blktype() != ASTBlock::BLK_EXCEPT || blk->nodes().empty())
        return false;
    PycRef<ASTCondBlock> condblk = blk.try_cast<ASTCondBlock>();
    if (condblk == NULL || condblk->cond() != NULL)
        return false;
    bool saw_bare_raise = false;
    for (const auto& child : blk->nodes()) {
        if (saw_bare_raise)
            break;
        if (child == NULL || is_pass_only_node(child))
            continue;
        PycRef<ASTRaise> raise = child.try_cast<ASTRaise>();
        if (raise != NULL && raise->params().empty()) {
            saw_bare_raise = true;
            continue;
        }
        return false;
    }
    return saw_bare_raise;
}

static bool extract_cond_terminal_name(const PycRef<ASTNode>& cond, std::string& out_name)
{
    if (cond == NULL)
        return false;
    if (cond.type() == ASTNode::NODE_NAME) {
        out_name = cond.cast<ASTName>()->name()->value();
        return true;
    }
    if (cond.type() == ASTNode::NODE_BINARY || cond.type() == ASTNode::NODE_COMPARE) {
        PycRef<ASTBinary> bin = cond.cast<ASTBinary>();
        if (bin->right() != NULL)
            return extract_cond_terminal_name(bin->right(), out_name);
    }
    return false;
}

static PycRef<ASTNode> build_none_tuple_return(int arity)
{
    if (arity <= 0)
        return new ASTReturn(new ASTObject(Pyc_None));
    ASTTuple::value_t vals;
    for (int i = 0; i < arity; ++i)
        vals.push_back(new ASTObject(Pyc_None));
    return new ASTReturn(new ASTTuple(vals));
}

static PycRef<ASTReturn> trailing_return_stmt(const PycRef<ASTBlock>& blk)
{
    if (blk == NULL || blk->nodes().empty())
        return NULL;
    return blk->nodes().back().try_cast<ASTReturn>();
}

static bool is_single_none_return_try(const PycRef<ASTBlock>& blk)
{
    if (blk == NULL || blk->blktype() != ASTBlock::BLK_TRY || blk->nodes().size() != 1)
        return false;
    PycRef<ASTReturn> ret = blk->nodes().front().try_cast<ASTReturn>();
    return ret != NULL && is_none_like_node(ret->value());
}

static void normalize_nested_with_except(ASTBlock::list_t& lines)
{
    for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTBlock> outer_try = (*it).try_cast<ASTBlock>();
        if (outer_try == NULL || outer_try->blktype() != ASTBlock::BLK_TRY)
            continue;

        ASTBlock::list_t::iterator inner_it = it;
        ++inner_it;
        if (inner_it == lines.end())
            continue;
        PycRef<ASTBlock> inner_ex = (*inner_it).try_cast<ASTBlock>();
        if (inner_ex == NULL || inner_ex->blktype() != ASTBlock::BLK_EXCEPT) {
            continue;
        }
        if (block_has_terminal_stmt_before_synthetic_except_tail(inner_ex))
            continue;

        ASTBlock::list_t::iterator cleanup_it = inner_it;
        ++cleanup_it;
        if (cleanup_it == lines.end())
            continue;
        PycRef<ASTBlock> cleanup_try = (*cleanup_it).try_cast<ASTBlock>();
        if (!is_single_none_return_try(cleanup_try))
            continue;

        ASTBlock::list_t::iterator outer_it = cleanup_it;
        ++outer_it;
        if (outer_it == lines.end())
            continue;
        PycRef<ASTBlock> outer_ex = (*outer_it).try_cast<ASTBlock>();
        if (outer_ex == NULL || outer_ex->blktype() != ASTBlock::BLK_EXCEPT
                || !block_returns_none_only(outer_ex)) {
            continue;
        }

        std::string inner_name;
        std::string outer_name;
        PycRef<ASTCondBlock> inner_cond = inner_ex.try_cast<ASTCondBlock>();
        PycRef<ASTCondBlock> outer_cond = outer_ex.try_cast<ASTCondBlock>();
        if (inner_cond == NULL || outer_cond == NULL
                || !extract_cond_terminal_name(inner_cond->cond(), inner_name)
                || !extract_cond_terminal_name(outer_cond->cond(), outer_name)
                || inner_name != outer_name) {
            continue;
        }

        PycRef<ASTReturn> shared_ret = trailing_return_stmt(outer_try);
        if (shared_ret == NULL || shared_ret->value() == NULL)
            continue;

        std::unordered_set<std::string> shared_names;
        collect_direct_store_names(inner_ex, shared_names);
        if (shared_names.empty() || !node_uses_any_name(shared_ret->value(), shared_names))
            continue;

        PycRef<ASTWithBlock> target_with = NULL;
        size_t split_idx = (size_t)-1;
        for (const auto& node : outer_try->nodes()) {
            PycRef<ASTWithBlock> withblk = node.try_cast<ASTWithBlock>();
            if (withblk == NULL)
                continue;

            size_t idx = 0;
            size_t last_store_idx = (size_t)-1;
            for (const auto& body_node : withblk->nodes()) {
                PycRef<ASTStore> st = body_node.try_cast<ASTStore>();
                if (st != NULL && store_dest_contains_name(st->dest(), shared_names))
                    last_store_idx = idx;
                ++idx;
            }

            if (last_store_idx != (size_t)-1) {
                target_with = withblk;
                split_idx = last_store_idx;
            }
        }
        if (target_with == NULL)
            continue;

        trim_synthetic_except_tail(inner_ex);

        ASTBlock::list_t prefix;
        ASTBlock::list_t suffix;
        size_t idx = 0;
        for (const auto& body_node : target_with->nodes()) {
            if (idx < split_idx)
                prefix.emplace_back(body_node);
            else
                suffix.emplace_back(body_node);
            ++idx;
        }
        if (suffix.empty())
            continue;

        PycRef<ASTBlock> nested_try = new ASTBlock(ASTBlock::BLK_TRY);
        nested_try->init();
        for (const auto& body_node : suffix)
            nested_try->append(body_node);

        while (target_with->size() > 0)
            target_with->removeLast();
        for (const auto& body_node : prefix)
            target_with->append(body_node);
        target_with->append(nested_try.cast<ASTNode>());

        target_with->append(inner_ex.cast<ASTNode>());

        if (shared_ret->value().type() == ASTNode::NODE_TUPLE) {
            int tuple_arity = (int)shared_ret->value().cast<ASTTuple>()->values().size();
            while (outer_ex->size() > 0)
                outer_ex->removeLast();
            outer_ex->append(build_none_tuple_return(tuple_arity));
        }

        lines.erase(cleanup_it);
        lines.erase(inner_it);
    }
}

struct MergedExceptionRegion {
    int start;
    int end;
    int target;
    int depth;
};

static bool compare_exception_regions(const MergedExceptionRegion& lhs, const MergedExceptionRegion& rhs)
{
    if (lhs.start != rhs.start)
        return lhs.start < rhs.start;
    if (lhs.end != rhs.end)
        return lhs.end < rhs.end;
    return lhs.target < rhs.target;
}

static std::vector<MergedExceptionRegion> merged_exception_regions(const PycRef<PycCode>& code, int depth)
{
    std::vector<MergedExceptionRegion> merged;
    if (code == NULL)
        return merged;

    std::unordered_map<int, size_t> by_target;
    for (const auto& entry : code->exceptionTableEntries()) {
        if (entry.stack_depth != depth)
            continue;
        auto it = by_target.find(entry.target);
        if (it == by_target.end()) {
            by_target.emplace(entry.target, merged.size());
            merged.push_back({ entry.start_offset, entry.end_offset, entry.target, depth });
        } else {
            MergedExceptionRegion& region = merged[it->second];
            region.start = std::min(region.start, entry.start_offset);
            region.end = std::max(region.end, entry.end_offset);
        }
    }

    std::sort(merged.begin(), merged.end(), compare_exception_regions);
    return merged;
}

struct ClassifiedExceptionRegions {
    std::vector<MergedExceptionRegion> top_level;
    std::vector<MergedExceptionRegion> nested;
};

static ClassifiedExceptionRegions classify_depth_zero_try_regions(const PycRef<PycCode>& code)
{
    ClassifiedExceptionRegions classified;
    if (code == NULL)
        return classified;

    int handler_boundary = std::numeric_limits<int>::max();
    bool has_entries = false;
    for (const auto& entry : code->exceptionTableEntries()) {
        handler_boundary = std::min(handler_boundary, entry.target);
        has_entries = true;
    }
    if (!has_entries)
        return classified;

    std::unordered_map<int, size_t> by_target;
    std::vector<MergedExceptionRegion> merged;
    for (const auto& entry : code->exceptionTableEntries()) {
        if (entry.stack_depth != 0 || entry.start_offset >= handler_boundary)
            continue;
        int end = std::min(entry.end_offset, handler_boundary);
        if (end <= entry.start_offset)
            continue;

        auto it = by_target.find(entry.target);
        if (it == by_target.end()) {
            by_target.emplace(entry.target, merged.size());
            merged.push_back({ entry.start_offset, end, entry.target, 0 });
        } else {
            MergedExceptionRegion& region = merged[it->second];
            region.start = std::min(region.start, entry.start_offset);
            region.end = std::max(region.end, end);
        }
    }

    std::sort(merged.begin(), merged.end(), compare_exception_regions);
    for (size_t i = 0; i < merged.size(); ++i) {
        bool contained = false;
        for (size_t j = 0; j < merged.size(); ++j) {
            if (i == j)
                continue;

            const MergedExceptionRegion& outer = merged[j];
            const MergedExceptionRegion& inner = merged[i];
            if (outer.start <= inner.start && inner.end <= outer.end
                    && (outer.start < inner.start || inner.end < outer.end)) {
                contained = true;
                break;
            }
        }

        if (contained)
            classified.nested.push_back(merged[i]);
        else
            classified.top_level.push_back(merged[i]);
    }
    return classified;
}

static std::vector<MergedExceptionRegion> filtered_top_level_try_regions(const PycRef<PycCode>& code)
{
    std::vector<MergedExceptionRegion> ordered;
    for (const auto& region : classify_depth_zero_try_regions(code).top_level) {
        if (!is_pass_handler_at_target(code, region.target, region.end))
            ordered.push_back(region);
    }
    return ordered;
}

static bool has_sequential_top_level_try_regions(const PycRef<PycCode>& code)
{
    std::vector<MergedExceptionRegion> ordered = filtered_top_level_try_regions(code);
    for (size_t i = 0; i + 1 < ordered.size(); ++i) {
        if (ordered[i].end <= ordered[i + 1].start)
            return true;
    }
    return false;
}

static size_t matching_suffix_len(const ASTBlock::list_t& lhs, const ASTBlock::list_t& rhs)
{
    size_t matched = 0;
    ASTBlock::list_t::const_reverse_iterator lit = lhs.rbegin();
    ASTBlock::list_t::const_reverse_iterator rit = rhs.rbegin();
    while (lit != lhs.rend() && rit != rhs.rend()) {
        if (node_signature(*lit) != node_signature(*rit))
            break;
        ++matched;
        ++lit;
        ++rit;
    }
    return matched;
}

static size_t matching_suffix_before_terminal_len(const ASTBlock::list_t& lhs,
        const ASTBlock::list_t& rhs)
{
    if (lhs.empty() || rhs.empty())
        return 0;

    ASTBlock::list_t::const_iterator lhs_end = lhs.end();
    if (is_terminal_stmt(lhs.back())) {
        --lhs_end;
        if (lhs.begin() == lhs_end)
            return 0;
    }

    size_t matched = 0;
    ASTBlock::list_t::const_reverse_iterator lit(lhs_end);
    ASTBlock::list_t::const_reverse_iterator rit = rhs.rbegin();
    while (lit != lhs.rend() && rit != rhs.rend()) {
        if (node_signature(*lit) != node_signature(*rit))
            break;
        ++matched;
        ++lit;
        ++rit;
    }
    return matched;
}

static void strip_block_suffix(const PycRef<ASTBlock>& blk, size_t count)
{
    while (blk != NULL && count-- > 0 && blk->size() > 0)
        blk->removeLast();
}

static void strip_block_suffix_before_terminal(const PycRef<ASTBlock>& blk, size_t count)
{
    if (blk == NULL || count == 0 || blk->size() == 0)
        return;
    if (!is_terminal_stmt(blk->nodes().back())) {
        strip_block_suffix(blk, count);
        return;
    }

    PycRef<ASTNode> terminal = blk->nodes().back();
    blk->removeLast();
    strip_block_suffix(blk, count);
    blk->append(terminal);
}

static void normalize_sequential_try_regions(ASTBlock::list_t& lines)
{
    if (!has_sequential_top_level_try_regions(cleanup_current_code))
        return;

    bool changed = true;
    while (changed) {
        changed = false;
        for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
            PycRef<ASTBlock> tryblk = (*it).try_cast<ASTBlock>();
            if (tryblk == NULL || tryblk->blktype() != ASTBlock::BLK_TRY)
                continue;

            ASTBlock::list_t::iterator ex_begin = it;
            ++ex_begin;
            ASTBlock::list_t::iterator ex_end = ex_begin;
            while (ex_end != lines.end()) {
                PycRef<ASTBlock> exblk = (*ex_end).try_cast<ASTBlock>();
                if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
                    break;
                ++ex_end;
            }
            if (ex_begin == ex_end)
                continue;

            size_t except_count = 0;
            for (ASTBlock::list_t::iterator cur = ex_begin; cur != ex_end; ++cur)
                ++except_count;
            if (except_count < 2)
                continue;

            ASTBlock::list_t::iterator outer_it = ex_end;
            --outer_it;
            PycRef<ASTBlock> outer_ex = (*outer_it).cast<ASTBlock>();
            if (!block_returns_none_only(outer_ex))
                continue;

            size_t best_suffix = 0;
            size_t matching_handlers = 0;
            for (ASTBlock::list_t::iterator cur = ex_begin; cur != outer_it; ++cur) {
                PycRef<ASTBlock> exblk = (*cur).cast<ASTBlock>();
                size_t suffix = matching_suffix_len(tryblk->nodes(), exblk->nodes());
                if (suffix > best_suffix)
                    best_suffix = suffix;
            }
            if (best_suffix < 2 || tryblk->nodes().size() <= best_suffix)
                continue;

            ASTBlock::list_t prefix;
            ASTBlock::list_t tail;
            size_t split_idx = tryblk->nodes().size() - best_suffix;
            size_t idx = 0;
            for (const auto& ln : tryblk->nodes()) {
                if (idx < split_idx)
                    prefix.emplace_back(ln);
                else
                    tail.emplace_back(ln);
                ++idx;
            }
            if (prefix.empty() || tail.empty())
                continue;

            for (ASTBlock::list_t::iterator cur = ex_begin; cur != outer_it; ++cur) {
                PycRef<ASTBlock> exblk = (*cur).cast<ASTBlock>();
                if (matching_suffix_len(tryblk->nodes(), exblk->nodes()) < best_suffix)
                    continue;
                strip_block_suffix(exblk, best_suffix);
                ++matching_handlers;
            }
            if (matching_handlers == 0)
                continue;

            while (tryblk->size() > 0)
                tryblk->removeLast();
            for (const auto& ln : prefix)
                tryblk->append(ln);

            PycRef<ASTBlock> tail_try = new ASTBlock(ASTBlock::BLK_TRY);
            tail_try->init();
            for (const auto& ln : tail)
                tail_try->append(ln);

            lines.insert(outer_it, tail_try.cast<ASTNode>());
            changed = true;
            break;
        }
    }
}

static void partition_nodes_by_offset_ranges(const ASTBlock::list_t& nodes,
        int first_end, int second_start,
        ASTBlock::list_t& first_region, ASTBlock::list_t& gap_region, ASTBlock::list_t& second_region)
{
    enum Bucket { FIRST, GAP, SECOND };
    Bucket bucket = FIRST;
    for (const auto& node : nodes) {
        int off = (node != NULL) ? node->offset() : -1;
        if (off >= 0) {
            if (off >= second_start)
                bucket = SECOND;
            else if (off >= first_end)
                bucket = GAP;
            else
                bucket = FIRST;
        }

        if (bucket == FIRST)
            first_region.emplace_back(node);
        else if (bucket == GAP)
            gap_region.emplace_back(node);
        else
            second_region.emplace_back(node);
    }
}

static void partition_nodes_by_exception_regions(const ASTBlock::list_t& nodes,
        int first_start, int first_end, int second_start,
        ASTBlock::list_t& prefix, ASTBlock::list_t& first_region,
        ASTBlock::list_t& gap_region, ASTBlock::list_t& second_region)
{
    enum Bucket { PREFIX, FIRST, GAP, SECOND };
    Bucket bucket = PREFIX;
    for (const auto& node : nodes) {
        int off = (node != NULL) ? node->offset() : -1;
        if (off >= 0) {
            if (off >= second_start)
                bucket = SECOND;
            else if (off >= first_end)
                bucket = GAP;
            else if (off >= first_start)
                bucket = FIRST;
            else
                bucket = PREFIX;
        }

        if (bucket == PREFIX)
            prefix.emplace_back(node);
        else if (bucket == FIRST)
            first_region.emplace_back(node);
        else if (bucket == GAP)
            gap_region.emplace_back(node);
        else
            second_region.emplace_back(node);
    }
}

static void partition_nodes_by_nested_range(const ASTBlock::list_t& nodes,
        int nested_start, int nested_end,
        ASTBlock::list_t& prefix, ASTBlock::list_t& nested, ASTBlock::list_t& suffix)
{
    enum Bucket { PREFIX, NESTED, SUFFIX };
    Bucket bucket = PREFIX;
    for (const auto& node : nodes) {
        int off = (node != NULL) ? node->offset() : -1;
        if (off >= 0) {
            if (off >= nested_end)
                bucket = SUFFIX;
            else if (off >= nested_start)
                bucket = NESTED;
            else
                bucket = PREFIX;
        }

        if (bucket == PREFIX) {
            PycRef<ASTBlock> blk = node.try_cast<ASTBlock>();
            if (blk != NULL
                    && blk->blktype() != ASTBlock::BLK_TRY
                    && blk->blktype() != ASTBlock::BLK_EXCEPT
                    && list_contains_offset_in_range(blk->nodes(), nested_start, nested_end))
                bucket = NESTED;
        }

        if (bucket == PREFIX)
            prefix.emplace_back(node);
        else if (bucket == NESTED)
            nested.emplace_back(node);
        else
            suffix.emplace_back(node);
    }
}

static bool find_nested_region_partition(const PycRef<ASTBlock>& blk,
        int nested_start, int nested_end, PycRef<ASTBlock>& owner,
        ASTBlock::list_t& prefix, ASTBlock::list_t& nested, ASTBlock::list_t& suffix)
{
    if (blk == NULL)
        return false;

    prefix.clear();
    nested.clear();
    suffix.clear();
    partition_nodes_by_nested_range(blk->nodes(), nested_start, nested_end,
            prefix, nested, suffix);
    if (!nested.empty()
            && !suffix.empty()
            && list_contains_offset_in_range(nested, nested_start, nested_end)) {
        owner = blk;
        return true;
    }

    for (const auto& node : blk->nodes()) {
        PycRef<ASTBlock> child = node.try_cast<ASTBlock>();
        if (child == NULL
                || child->blktype() == ASTBlock::BLK_TRY
                || child->blktype() == ASTBlock::BLK_EXCEPT
                || child->blktype() == ASTBlock::BLK_FINALLY
                || child->blktype() == ASTBlock::BLK_CONTAINER
                || !list_contains_offset_in_range(child->nodes(), nested_start, nested_end))
            continue;
        if (find_nested_region_partition(child, nested_start, nested_end,
                owner, prefix, nested, suffix))
            return true;
    }
    return false;
}

static int first_meaningful_offset_in_block(const PycRef<ASTBlock>& blk)
{
    if (blk == NULL)
        return -1;
    for (const auto& node : blk->nodes()) {
        if (node == NULL || is_pass_only_node(node))
            continue;
        if (node->offset() >= 0)
            return node->offset();
        PycRef<ASTBlock> nested = node.try_cast<ASTBlock>();
        if (nested != NULL) {
            int nested_off = first_meaningful_offset_in_block(nested);
            if (nested_off >= 0)
                return nested_off;
        }
    }
    return -1;
}

static bool list_contains_offset_in_range(const ASTBlock::list_t& nodes, int start, int end)
{
    for (const auto& node : nodes) {
        if (node == NULL)
            continue;
        int off = node->offset();
        if (off >= start && off < end)
            return true;
        PycRef<ASTBlock> nested = node.try_cast<ASTBlock>();
        if (nested != NULL && list_contains_offset_in_range(nested->nodes(), start, end))
            return true;
    }
    return false;
}

static bool is_pass_handler_at_target(const PycRef<PycCode>& code, int target,
        int continuation, int max_stack_depth)
{
    if (code == NULL || cleanup_current_module == NULL || code->code() == NULL
            || target < 0 || target >= code->code()->length())
        return false;

    bool has_target = false;
    int scan_limit = code->code()->length();
    for (const auto& entry : code->exceptionTableEntries()) {
        if (entry.stack_depth > max_stack_depth)
            continue;
        if (entry.target == target) {
            has_target = true;
            continue;
        }
        if (entry.target > target)
            scan_limit = std::min(scan_limit, entry.target);
    }
    if (!has_target)
        return false;

    const unsigned char* base = (const unsigned char*)code->code()->value();
    PycBuffer source(base + target, code->code()->length() - target);
    int cursor = target;
    bool saw_match = false;
    bool entered_body = false;
    bool saw_pop_except = false;
    int alias_store_opcode = -1;
    int alias_store_operand = -1;
    int alias_cleanup_stage = 0;

    while (!source.atEof() && cursor < scan_limit) {
        int opcode, operand;
        const int inst_start = cursor;
        bc_next(source, cleanup_current_module, opcode, operand, cursor);
        if (is_handler_scan_noop(opcode))
            continue;

        if (!saw_pop_except) {
            if (opcode == Pyc::CHECK_EXC_MATCH) {
                saw_match = true;
                continue;
            }
            if (!entered_body) {
                if (opcode == Pyc::POP_TOP) {
                    entered_body = true;
                    continue;
                }
                if (saw_match && is_exception_alias_store_opcode(opcode)) {
                    entered_body = true;
                    alias_store_opcode = opcode;
                    alias_store_operand = operand;
                    continue;
                }
                continue;
            }
            if (opcode == Pyc::POP_EXCEPT) {
                saw_pop_except = true;
                continue;
            }
            return false;
        }

        if (continuation_matches_from(code, inst_start,
                next_meaningful_opcode_pos(code, cleanup_current_module, continuation),
                scan_limit))
            return true;

        if (alias_store_opcode != -1) {
            if (alias_cleanup_stage == 0 && is_none_load_const(code, opcode, operand)) {
                alias_cleanup_stage = 1;
                continue;
            }
            if (alias_cleanup_stage == 1
                    && opcode == alias_store_opcode && operand == alias_store_operand) {
                alias_cleanup_stage = 2;
                continue;
            }
            if (alias_cleanup_stage == 2
                    && is_exception_alias_delete_opcode(opcode)
                    && operand == alias_store_operand) {
                alias_cleanup_stage = 3;
                continue;
            }
        }

        if (opcode == Pyc::RETURN_VALUE
                || opcode == Pyc::RETURN_CONST_A
                || opcode == Pyc::INSTRUMENTED_RETURN_VALUE_A
                || opcode == Pyc::INSTRUMENTED_RETURN_CONST_A)
            return false;
        if (is_jump_opcode(opcode)
                || opcode == Pyc::COPY_A
                || opcode == Pyc::RERAISE
                || opcode == Pyc::RERAISE_A)
            return true;
        return false;
    }
    return false;
}

static ASTBlock::list_t::iterator find_matching_typed_except_block(ASTBlock::list_t& lines,
        ASTBlock::list_t::iterator try_it, int target)
{
    ASTBlock::list_t::iterator best = lines.end();
    int best_offset = std::numeric_limits<int>::max();
    ASTBlock::list_t::iterator ex_it = try_it;
    ++ex_it;
    for (; ex_it != lines.end(); ++ex_it) {
        PycRef<ASTBlock> exblk = (*ex_it).try_cast<ASTBlock>();
        if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
            break;

        PycRef<ASTCondBlock> excond = exblk.try_cast<ASTCondBlock>();
        if (excond == NULL || excond->cond() == NULL)
            continue;

        int first_offset = first_meaningful_offset_in_block(exblk);
        if (first_offset < 0)
            first_offset = exblk->offset();
        if (first_offset < target || first_offset >= best_offset)
            continue;

        best = ex_it;
        best_offset = first_offset;
    }
    return best;
}

static bool is_bare_reraise_node(const PycRef<ASTNode>& node)
{
    PycRef<ASTRaise> raise = node.try_cast<ASTRaise>();
    return raise != NULL && raise->params().empty();
}

static void partition_except_blocks_by_target(ASTBlock::list_t& lines,
        ASTBlock::list_t::iterator begin, ASTBlock::list_t::iterator end,
        int second_target,
        std::vector<ASTBlock::list_t::iterator>& first_handlers,
        std::vector<ASTBlock::list_t::iterator>& second_handlers)
{
    bool seen_second = false;
    for (ASTBlock::list_t::iterator it = begin; it != end; ++it) {
        PycRef<ASTBlock> exblk = (*it).try_cast<ASTBlock>();
        if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
            continue;

        int first_offset = first_meaningful_offset_in_block(exblk);
        if (first_offset < 0)
            first_offset = exblk->offset();
        if (!seen_second && first_offset >= second_target)
            seen_second = true;

        if (seen_second)
            second_handlers.push_back(it);
        else
            first_handlers.push_back(it);
    }
}

static void normalize_exception_table_partitions(ASTBlock::list_t& lines)
{
    ClassifiedExceptionRegions classified = classify_depth_zero_try_regions(cleanup_current_code);
    const std::vector<MergedExceptionRegion>& top_regions = classified.top_level;
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t top_idx = 0; top_idx + 1 < top_regions.size() && !changed; ++top_idx) {
            const MergedExceptionRegion& first_top = top_regions[top_idx];
            const MergedExceptionRegion& second_top = top_regions[top_idx + 1];
            if (first_top.end > second_top.start)
                continue;

            MergedExceptionRegion nested_region = { 0, 0, 0, 1 };
            bool has_nested_in_second = false;
            for (const auto& region : classified.nested) {
                if (region.start >= second_top.start && region.end <= second_top.end) {
                    nested_region = region;
                    has_nested_in_second = true;
                    break;
                }
            }

            for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
                PycRef<ASTBlock> tryblk = (*it).try_cast<ASTBlock>();
                if (tryblk == NULL || tryblk->blktype() != ASTBlock::BLK_TRY)
                    continue;

                ASTBlock::list_t::iterator ex1_it = it;
                ++ex1_it;
                if (ex1_it == lines.end())
                    continue;
                PycRef<ASTBlock> ex1 = (*ex1_it).try_cast<ASTBlock>();
                if (ex1 == NULL || ex1->blktype() != ASTBlock::BLK_EXCEPT)
                    continue;

                ASTBlock::list_t::iterator ex2_it = ex1_it;
                ASTBlock::list_t::iterator ex3_it = ex2_it;
                PycRef<ASTBlock> ex2;
                PycRef<ASTBlock> ex3;
                bool split_nested_handler_siblings = false;
                bool has_nested = has_nested_in_second;
                if (has_nested) {
                    if (!block_returns_none_only(ex1))
                        has_nested = false;
                    else {
                        ++ex2_it;
                        if (ex2_it != lines.end()) {
                            ex2 = (*ex2_it).try_cast<ASTBlock>();
                            if (ex2 != NULL && ex2->blktype() == ASTBlock::BLK_EXCEPT) {
                                ex3_it = ex2_it;
                                ++ex3_it;
                                if (ex3_it != lines.end()) {
                                    ex3 = (*ex3_it).try_cast<ASTBlock>();
                                    if (ex3 != NULL && ex3->blktype() == ASTBlock::BLK_EXCEPT)
                                        split_nested_handler_siblings = true;
                                }
                            }
                        }
                        if (!split_nested_handler_siblings)
                            has_nested = false;
                    }
                }

                ASTBlock::list_t prefix_region;
                ASTBlock::list_t first_region;
                ASTBlock::list_t gap_region;
                ASTBlock::list_t second_region;
                partition_nodes_by_exception_regions(tryblk->nodes(),
                        first_top.start, first_top.end, second_top.start,
                        prefix_region, first_region, gap_region, second_region);
                if (first_region.empty() || second_region.empty())
                    continue;
                bool prefix_contains_intermediate_region = false;
                for (const auto& region : top_regions) {
                    if ((region.start == first_top.start && region.end == first_top.end
                                && region.target == first_top.target)
                            || (region.start == second_top.start && region.end == second_top.end
                                && region.target == second_top.target)) {
                        continue;
                    }
                    if (list_contains_offset_in_range(prefix_region, region.start, region.end)) {
                        prefix_contains_intermediate_region = true;
                        break;
                    }
                }
                if (prefix_contains_intermediate_region)
                    continue;

                if (!has_nested) {
                    ASTBlock::list_t::iterator after_ex_it = ex1_it;
                    for (; after_ex_it != lines.end(); ++after_ex_it) {
                        PycRef<ASTBlock> exblk = (*after_ex_it).try_cast<ASTBlock>();
                        if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
                            break;
                    }

                    std::vector<ASTBlock::list_t::iterator> first_handlers;
                    std::vector<ASTBlock::list_t::iterator> second_handlers;
                    partition_except_blocks_by_target(lines, ex1_it, after_ex_it,
                            second_top.target, first_handlers, second_handlers);
                    if (first_handlers.empty() || second_handlers.empty())
                        continue;

                    std::vector<PycRef<ASTNode>> second_handler_nodes;
                    for (const auto& handler_it : second_handlers)
                        second_handler_nodes.push_back(*handler_it);

                    while (tryblk->size() > 0)
                        tryblk->removeLast();
                    for (const auto& node : first_region)
                        tryblk->append(node);
                    for (const auto& node : prefix_region)
                        lines.insert(it, node);

                    for (const auto& handler_it : second_handlers)
                        lines.erase(handler_it);

                    ASTBlock::list_t::iterator insert_it = after_ex_it;
                    for (const auto& node : gap_region)
                        lines.insert(insert_it, node);

                    PycRef<ASTBlock> tail_try = new ASTBlock(ASTBlock::BLK_TRY);
                    tail_try->init();
                    for (const auto& node : second_region)
                        tail_try->append(node);
                    lines.insert(insert_it, tail_try.cast<ASTNode>());

                    for (const auto& node : second_handler_nodes)
                        lines.insert(insert_it, node);
                    changed = true;
                    break;
                }

                ASTBlock::list_t second_prefix;
                ASTBlock::list_t nested_body;
                ASTBlock::list_t second_suffix;
                partition_nodes_by_nested_range(second_region,
                        nested_region.start, nested_region.end,
                        second_prefix, nested_body, second_suffix);
                if (nested_body.empty())
                    continue;

                while (tryblk->size() > 0)
                    tryblk->removeLast();
                for (const auto& node : first_region)
                    tryblk->append(node);
                for (const auto& node : prefix_region)
                    lines.insert(it, node);

                PycRef<ASTBlock> second_try = new ASTBlock(ASTBlock::BLK_TRY);
                second_try->init();
                for (const auto& node : second_prefix)
                    second_try->append(node);

                PycRef<ASTBlock> nested_try = new ASTBlock(ASTBlock::BLK_TRY);
                nested_try->init();
                for (const auto& node : nested_body)
                    nested_try->append(node);
                second_try->append(nested_try.cast<ASTNode>());
                second_try->append(ex2.cast<ASTNode>());
                for (const auto& node : second_suffix)
                    second_try->append(node);

                for (const auto& node : gap_region)
                    lines.insert(ex2_it, node);
                lines.insert(ex3_it, second_try.cast<ASTNode>());
                lines.erase(ex2_it);
                changed = true;
                break;
            }
        }
    }

    for (const auto& region : classified.nested) {
        for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
            PycRef<ASTBlock> tryblk = (*it).try_cast<ASTBlock>();
            if (tryblk == NULL || tryblk->blktype() != ASTBlock::BLK_TRY)
                continue;

            PycRef<ASTBlock> owner = tryblk;
            ASTBlock::list_t prefix;
            ASTBlock::list_t nested_body;
            ASTBlock::list_t suffix;
            if (!find_nested_region_partition(tryblk, region.start, region.end,
                    owner, prefix, nested_body, suffix))
                continue;
            if (nested_body.empty()
                    || suffix.empty()
                    || !list_contains_offset_in_range(nested_body, region.start, region.end))
                continue;

            ASTBlock::list_t::iterator nested_ex_it = find_matching_typed_except_block(lines, it, region.target);
            if (nested_ex_it == lines.end())
                continue;

            PycRef<ASTBlock> nested_ex = (*nested_ex_it).cast<ASTBlock>();
            PycRef<ASTBlock> nested_try = new ASTBlock(ASTBlock::BLK_TRY);
            nested_try->init();
            for (const auto& node : nested_body)
                nested_try->append(node);

            while (owner->size() > 0)
                owner->removeLast();
            for (const auto& node : prefix)
                owner->append(node);
            owner->append(nested_try.cast<ASTNode>());
            owner->append(nested_ex.cast<ASTNode>());
            for (const auto& node : suffix)
                owner->append(node);

            lines.erase(nested_ex_it);
            break;
        }
    }
}

static void normalize_nested_except_handler_partitions(ASTBlock::list_t& lines)
{
    std::vector<MergedExceptionRegion> nested_regions = merged_exception_regions(cleanup_current_code, 1);
    if (nested_regions.empty())
        return;

    for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTBlock> tryblk = (*it).try_cast<ASTBlock>();
        if (tryblk == NULL || tryblk->blktype() != ASTBlock::BLK_TRY)
            continue;

        ASTBlock::list_t::iterator ex_it = it;
        ++ex_it;
        for (; ex_it != lines.end(); ++ex_it) {
            PycRef<ASTBlock> exblk = (*ex_it).try_cast<ASTBlock>();
            if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
                break;

            PycRef<ASTCondBlock> excond = exblk.try_cast<ASTCondBlock>();
            if (excond == NULL || excond->cond() == NULL)
                continue;

            for (const auto& region : nested_regions) {
                PycRef<ASTBlock> owner = exblk;
                ASTBlock::list_t prefix;
                ASTBlock::list_t nested_body;
                ASTBlock::list_t suffix;
                if (!find_nested_region_partition(exblk, region.start, region.end,
                        owner, prefix, nested_body, suffix))
                    continue;
                if (nested_body.empty() || suffix.empty())
                    continue;

                ASTBlock::list_t nested_handlers;
                ASTBlock::list_t tail;
                bool collecting_handlers = true;
                for (const auto& node : suffix) {
                    PycRef<ASTBlock> child = node.try_cast<ASTBlock>();
                    if (collecting_handlers && child != NULL && child->blktype() == ASTBlock::BLK_EXCEPT) {
                        nested_handlers.emplace_back(node);
                        continue;
                    }
                    collecting_handlers = false;
                    tail.emplace_back(node);
                }
                if (nested_handlers.empty())
                    continue;

                PycRef<ASTBlock> nested_try = new ASTBlock(ASTBlock::BLK_TRY);
                nested_try->init();
                for (const auto& node : nested_body)
                    nested_try->append(node);

                while (owner->size() > 0)
                    owner->removeLast();
                for (const auto& node : prefix)
                    owner->append(node);
                owner->append(nested_try.cast<ASTNode>());
                for (const auto& node : nested_handlers)
                    owner->append(node);
                for (const auto& node : tail)
                    owner->append(node);
                break;
            }
        }
    }
}

static void normalize_trailing_reraise_finally(ASTBlock::list_t& lines)
{
    for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTBlock> tryblk = (*it).try_cast<ASTBlock>();
        if (tryblk == NULL || tryblk->blktype() != ASTBlock::BLK_TRY)
            continue;

        ASTBlock::list_t::iterator ex_begin = it;
        ++ex_begin;
        ASTBlock::list_t::iterator ex_end = ex_begin;
        while (ex_end != lines.end()) {
            PycRef<ASTBlock> exblk = (*ex_end).try_cast<ASTBlock>();
            if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT)
                break;
            ++ex_end;
        }
        if (ex_begin == ex_end)
            continue;

        // Scan the tail past try+except chains (not just non-except nodes).
        // Python 3.11+ duplicates finally blocks: once for the normal path
        // and once for the exception path, ending with a bare reraise.
        // The tail may contain: [try, except*, try, except*, except:raise]
        ASTBlock::list_t::iterator tail_end = ex_end;
        while (tail_end != lines.end()) {
            PycRef<ASTBlock> tail_blk = (*tail_end).try_cast<ASTBlock>();
            if (tail_blk != NULL
                    && (tail_blk->blktype() == ASTBlock::BLK_FINALLY
                        || tail_blk->blktype() == ASTBlock::BLK_CONTAINER))
                break;
            ++tail_end;
        }
        if (ex_end == tail_end) {
            continue;
        }

        ASTBlock::list_t tail_nodes;
        for (ASTBlock::list_t::iterator tail_it = ex_end; tail_it != tail_end; ++tail_it)
            tail_nodes.emplace_back(*tail_it);
        while (tail_nodes.size() > 1
                && (is_pass_only_node(tail_nodes.back())
                    || is_synthetic_loop_keyword_node(tail_nodes.back())))
            tail_nodes.pop_back();

        // Check for bare reraise: either a standalone raise or an except block
        // containing only a bare raise (except: raise).
        bool found_reraise = false;
        if (!tail_nodes.empty()) {
            if (is_bare_reraise_node(tail_nodes.back())) {
                found_reraise = true;
            } else {
                PycRef<ASTBlock> last_blk = tail_nodes.back().try_cast<ASTBlock>();
                if (last_blk != NULL && is_bare_reraise_except_block(last_blk))
                    found_reraise = true;
            }
        }
        if (tail_nodes.size() < 2 || !found_reraise)
            continue;

        tail_nodes.pop_back();

        // Deduplicate the normal-path and exception-path finally copies.
        // Parse tail_nodes into groups: [try_a, except_a*, ..., try_b, except_b*]
        // try_a = normal-path copy (full, may include extra statements and terminal)
        // try_b = exception-path copy (minimal cleanup only)
        PycRef<ASTBlock> first_try, second_try;
        ASTBlock::list_t first_excepts, second_excepts;
        ASTBlock::list_t gap_nodes;
        int parse_phase = 0;

        for (const auto& node : tail_nodes) {
            PycRef<ASTBlock> blk = node.try_cast<ASTBlock>();
            if (blk != NULL && blk->blktype() == ASTBlock::BLK_TRY) {
                if (first_try == NULL) {
                    first_try = blk;
                    parse_phase = 1;
                } else if (second_try == NULL) {
                    second_try = blk;
                    parse_phase = 3;
                }
                continue;
            }
            if (blk != NULL && blk->blktype() == ASTBlock::BLK_EXCEPT) {
                if (parse_phase == 1) { first_excepts.emplace_back(node); continue; }
                if (parse_phase == 3) { second_excepts.emplace_back(node); continue; }
            }
            if (parse_phase == 1) parse_phase = 2;
            if (parse_phase <= 2) gap_nodes.emplace_back(node);
        }

        ASTBlock::list_t finally_body;
        if (second_try != NULL && first_try != NULL) {
            // Two copies found. Use the second (minimal) as the core try block.
            // Extract extra statements from the first try that go beyond
            // the second try's content, excluding terminal statements.
            size_t matching_prefix = 0;
            auto it1 = first_try->nodes().begin();
            auto it2 = second_try->nodes().begin();
            while (it1 != first_try->nodes().end() && it2 != second_try->nodes().end()) {
                if (node_signature(*it1) != node_signature(*it2))
                    break;
                ++matching_prefix;
                ++it1;
                ++it2;
            }

            if (matching_prefix == second_try->size() && matching_prefix <= first_try->size()) {
                // second_try is a prefix of first_try.
                // Core finally: second_try + second_excepts
                // Extra: first_try nodes after the prefix, minus terminal
                finally_body.emplace_back(second_try.cast<ASTNode>());
                for (const auto& ex : second_excepts)
                    finally_body.emplace_back(ex);
                for (; it1 != first_try->nodes().end(); ++it1) {
                    if (!is_terminal_stmt(*it1))
                        finally_body.emplace_back(*it1);
                }
            } else {
                // Fallback: use first try stripped of terminal + its excepts
                while (first_try->size() > 0 && is_terminal_stmt(first_try->nodes().back()))
                    first_try->removeLast();
                finally_body.emplace_back(first_try.cast<ASTNode>());
                for (const auto& ex : first_excepts)
                    finally_body.emplace_back(ex);
            }
        } else if (first_try != NULL) {
            // Single try copy. Strip terminal and use it.
            while (first_try->size() > 0 && is_terminal_stmt(first_try->nodes().back()))
                first_try->removeLast();
            finally_body.emplace_back(first_try.cast<ASTNode>());
            for (const auto& ex : first_excepts)
                finally_body.emplace_back(ex);
        } else {
            // No try blocks in tail - use tail_nodes as-is (original behavior)
            finally_body = tail_nodes;
        }

        PycRef<ASTBlock> finallyblk = new ASTBlock(ASTBlock::BLK_FINALLY, 0, true);
        finallyblk->init();
        for (const auto& node : finally_body)
            finallyblk->append(node);
        if (!finally_body.empty()) {
            size_t direct_suffix = matching_suffix_len(tryblk->nodes(), finally_body);
            if (direct_suffix == finally_body.size()) {
                strip_block_suffix(tryblk, direct_suffix);
            } else {
                size_t terminal_suffix = matching_suffix_before_terminal_len(tryblk->nodes(), finally_body);
                if (terminal_suffix == finally_body.size())
                    strip_block_suffix_before_terminal(tryblk, terminal_suffix);
            }
        }

        ASTBlock::list_t::iterator insert_pos = lines.erase(ex_end, tail_end);
        lines.insert(insert_pos, finallyblk.cast<ASTNode>());
    }
}

static bool is_logical_condition_node(const PycRef<ASTNode>& node)
{
    if (node == NULL)
        return false;
    if (node.type() != ASTNode::NODE_BINARY && node.type() != ASTNode::NODE_COMPARE)
        return false;

    PycRef<ASTBinary> bin = node.cast<ASTBinary>();
    return bin->op() == ASTBinary::BIN_LOG_AND || bin->op() == ASTBinary::BIN_LOG_OR;
}

static void print_condition_expr(const PycRef<ASTNode>& cond, bool negative,
        PycModule* mod, std::ostream& pyc_output, bool padded)
{
    if (negative) {
        if (padded)
            pyc_output << " not";
        else
            pyc_output << "not";

        if (is_logical_condition_node(cond)) {
            pyc_output << " (";
            print_src(cond, mod, pyc_output);
            pyc_output << ")";
        } else {
            pyc_output << " ";
            print_src(cond, mod, pyc_output);
        }
    } else {
        if (padded)
            pyc_output << " ";
        print_src(cond, mod, pyc_output);
    }
}

static void normalize_except_return_rejoin(ASTBlock::list_t& lines)
{
    for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTBlock> exblk = (*it).try_cast<ASTBlock>();
        if (exblk == NULL || exblk->blktype() != ASTBlock::BLK_EXCEPT || block_has_terminal_stmt(exblk))
            continue;
        if (it == lines.begin())
            continue;

        ASTBlock::list_t::iterator prev_it = it;
        --prev_it;
        PycRef<ASTReturn> shared_ret = (*prev_it).try_cast<ASTReturn>();
        if (shared_ret == NULL || shared_ret->value() == NULL)
            continue;
        if (prev_it != lines.begin()) {
            ASTBlock::list_t::iterator before_prev = prev_it;
            --before_prev;
            if (is_none_exit_call_any_receiver(*before_prev))
                continue;
        }

        ASTBlock::list_t::iterator try_it = it;
        ++try_it;
        if (try_it == lines.end())
            continue;
        PycRef<ASTBlock> tryblk = (*try_it).try_cast<ASTBlock>();
        if (tryblk == NULL || tryblk->blktype() != ASTBlock::BLK_TRY
                || tryblk->nodes().size() != 1) {
            continue;
        }
        PycRef<ASTReturn> try_ret = tryblk->nodes().front().try_cast<ASTReturn>();
        if (try_ret == NULL || !is_none_like_node(try_ret->value()))
            continue;

        ASTBlock::list_t::iterator outer_it = try_it;
        ++outer_it;
        if (outer_it == lines.end())
            continue;
        PycRef<ASTBlock> outer_ex = (*outer_it).try_cast<ASTBlock>();
        if (outer_ex == NULL || outer_ex->blktype() != ASTBlock::BLK_EXCEPT || !block_returns_none_only(outer_ex))
            continue;

        std::string inner_name;
        std::string outer_name;
        PycRef<ASTCondBlock> inner_cond = exblk.try_cast<ASTCondBlock>();
        PycRef<ASTCondBlock> outer_cond = outer_ex.try_cast<ASTCondBlock>();
        if (inner_cond == NULL || outer_cond == NULL
                || !extract_cond_terminal_name(inner_cond->cond(), inner_name)
                || !extract_cond_terminal_name(outer_cond->cond(), outer_name)
                || inner_name != outer_name) {
            continue;
        }

        while (exblk->size() > 0 && is_pass_only_node(exblk->nodes().back()))
            exblk->removeLast();
        exblk->append(new ASTReturn(shared_ret->value()));

        if (shared_ret->value().type() == ASTNode::NODE_TUPLE) {
            int tuple_arity = (int)shared_ret->value().cast<ASTTuple>()->values().size();
            while (outer_ex->size() > 0)
                outer_ex->removeLast();
            outer_ex->append(build_none_tuple_return(tuple_arity));
        }

        lines.erase(try_it);
    }
}

static PycRef<ASTBlock> trailing_block_node(PycRef<ASTNode> node)
{
    if (node == NULL)
        return NULL;
    if (node.type() == ASTNode::NODE_BLOCK)
        return node.cast<ASTBlock>();
    if (node.type() == ASTNode::NODE_NODELIST) {
        const ASTNodeList::list_t& nodes = node.cast<ASTNodeList>()->nodes();
        for (ASTNodeList::list_t::const_reverse_iterator rit = nodes.rbegin();
                rit != nodes.rend(); ++rit) {
            if ((*rit) == NULL)
                continue;
            PycRef<ASTBlock> nested = trailing_block_node(*rit);
            if (nested != NULL)
                return nested;
            break;
        }
    }
    return NULL;
}

static bool cond_ends_with_name(const PycRef<ASTNode>& cond, const char* name)
{
    if (cond == NULL || name == NULL)
        return false;
    if (cond.type() == ASTNode::NODE_NAME)
        return cond.cast<ASTName>()->name()->isEqual(name);
    if (cond.type() == ASTNode::NODE_BINARY || cond.type() == ASTNode::NODE_COMPARE) {
        PycRef<ASTBinary> bin = cond.cast<ASTBinary>();
        if (bin->op() == ASTBinary::BIN_ATTR
                && bin->right() != NULL
                && bin->right().type() == ASTNode::NODE_NAME
                && bin->right().cast<ASTName>()->name()->isEqual(name)) {
            return true;
        }
        return cond_ends_with_name(bin->right(), name);
    }
    return false;
}

static bool block_has_terminal_stmt(const PycRef<ASTBlock>& blk)
{
    if (blk == NULL || blk->nodes().empty())
        return false;
    return is_terminal_stmt(blk->nodes().back());
}

static bool line_has_tuple_return_arity(const PycRef<ASTNode>& node, int arity)
{
    if (node == NULL)
        return false;
    if (node.type() == ASTNode::NODE_RETURN) {
        PycRef<ASTNode> value = node.cast<ASTReturn>()->value();
        if (value != NULL && value.type() == ASTNode::NODE_TUPLE
                && (int)value.cast<ASTTuple>()->values().size() == arity) {
            return true;
        }
        return false;
    }
    if (node.type() == ASTNode::NODE_BLOCK) {
        for (const auto& child : node.cast<ASTBlock>()->nodes()) {
            if (line_has_tuple_return_arity(child, arity))
                return true;
        }
        return false;
    }
    if (node.type() == ASTNode::NODE_NODELIST) {
        for (const auto& child : node.cast<ASTNodeList>()->nodes()) {
            if (line_has_tuple_return_arity(child, arity))
                return true;
        }
        return false;
    }
    return false;
}

static bool list_has_tuple_return_arity(const ASTBlock::list_t& lines, int arity)
{
    for (const auto& node : lines) {
        if (line_has_tuple_return_arity(node, arity))
            return true;
    }
    return false;
}

static bool line_has_bool_tuple_return(const PycRef<ASTNode>& node)
{
    if (node == NULL)
        return false;
    if (node.type() == ASTNode::NODE_RETURN) {
        PycRef<ASTNode> value = node.cast<ASTReturn>()->value();
        if (value == NULL || value.type() != ASTNode::NODE_TUPLE)
            return false;
        const ASTTuple::value_t& vals = value.cast<ASTTuple>()->values();
        if (vals.size() != 2 || vals[0] == NULL || vals[0].type() != ASTNode::NODE_OBJECT)
            return false;
        PycRef<PycObject> first = vals[0].cast<ASTObject>()->object();
        return first == Pyc_True || first == Pyc_False
                || first->type() == PycObject::TYPE_TRUE
                || first->type() == PycObject::TYPE_FALSE;
    }
    if (node.type() == ASTNode::NODE_BLOCK) {
        for (const auto& child : node.cast<ASTBlock>()->nodes()) {
            if (line_has_bool_tuple_return(child))
                return true;
        }
        return false;
    }
    if (node.type() == ASTNode::NODE_NODELIST) {
        for (const auto& child : node.cast<ASTNodeList>()->nodes()) {
            if (line_has_bool_tuple_return(child))
                return true;
        }
        return false;
    }
    return false;
}

static bool block_returns_none_only(const PycRef<ASTBlock>& blk)
{
    if (blk == NULL || blk->nodes().empty())
        return false;
    PycRef<ASTReturn> ret = blk->nodes().front().try_cast<ASTReturn>();
    if (ret == NULL || !is_all_none_like_node(ret->value()))
        return false;
    ASTBlock::list_t::const_iterator it = blk->nodes().begin();
    ++it;
    for (; it != blk->nodes().end(); ++it) {
        if ((*it) == NULL || (*it).type() != ASTNode::NODE_RAISE)
            return false;
    }
    return true;
}

static ASTCall::pparam_t normalize_exit_none_params(const ASTCall::pparam_t& params)
{
    if (params.size() == 2) {
        bool all_none = true;
        for (const auto& p : params) {
            if (!is_none_like_node(p)) {
                all_none = false;
                break;
            }
        }
        if (all_none) {
            ASTCall::pparam_t out = params;
            out.push_back(new ASTObject(Pyc_None));
            return out;
        }
    }
    return params;
}

static bool is_empty_list_node(const PycRef<ASTNode>& node)
{
    return node != NULL && node.type() == ASTNode::NODE_LIST
            && node.cast<ASTList>()->values().empty();
}

static void rewrite_for_append_target(const PycRef<ASTBlock>& forblk, const PycRef<PycString>& target)
{
    if (forblk == NULL || target == NULL)
        return;
    ASTBlock::list_t rewritten;
    for (const auto& child : forblk->nodes()) {
        PycRef<ASTNode> out = child;
        if (child != NULL && child.type() == ASTNode::NODE_CALL) {
            PycRef<ASTCall> call = child.cast<ASTCall>();
            if (call->func() != NULL && call->func().type() == ASTNode::NODE_BINARY) {
                PycRef<ASTBinary> bin = call->func().cast<ASTBinary>();
                if (bin->op() == ASTBinary::BIN_ATTR
                        && is_empty_list_node(bin->left())
                        && bin->right() != NULL
                        && bin->right().type() == ASTNode::NODE_NAME
                        && bin->right().cast<ASTName>()->name()->isEqual("append")) {
                    out = new ASTCall(
                        new ASTBinary(new ASTName(target), new ASTName(bin->right().cast<ASTName>()->name()),
                                      ASTBinary::BIN_ATTR),
                        call->pparams(), call->kwparams());
                }
            }
        }
        rewritten.emplace_back(out);
    }
    while (forblk->size() > 0)
        forblk->removeLast();
    for (const auto& child : rewritten)
        forblk->append(child);
}

static bool extract_enter_assignment(const PycRef<ASTNode>& node,
                                     PycRef<PycString>& out_var,
                                     PycRef<ASTNode>& out_expr)
{
    if (node == NULL || node.type() != ASTNode::NODE_STORE)
        return false;
    PycRef<ASTStore> st = node.cast<ASTStore>();
    if (st->dest() == NULL || st->dest().type() != ASTNode::NODE_NAME
            || st->src() == NULL || st->src().type() != ASTNode::NODE_CALL)
        return false;

    PycRef<ASTCall> call = st->src().cast<ASTCall>();
    if (call->func() == NULL || call->func().type() != ASTNode::NODE_BINARY)
        return false;
    PycRef<ASTBinary> bin = call->func().cast<ASTBinary>();
    if (bin->op() != ASTBinary::BIN_ATTR
            || bin->right() == NULL
            || bin->right().type() != ASTNode::NODE_NAME
            || !bin->right().cast<ASTName>()->name()->isEqual("__enter__")) {
        return false;
    }

    out_var = st->dest().cast<ASTName>()->name();
    out_expr = bin->left();
    return true;
}

static bool is_exit_call_for_var(const PycRef<ASTNode>& node, const PycRef<PycString>& var_name)
{
    if (node == NULL || node.type() != ASTNode::NODE_CALL)
        return false;
    PycRef<ASTCall> call = node.cast<ASTCall>();
    if (call->func() == NULL || call->func().type() != ASTNode::NODE_BINARY)
        return false;
    PycRef<ASTBinary> bin = call->func().cast<ASTBinary>();
    if (bin->op() != ASTBinary::BIN_ATTR
            || bin->right() == NULL
            || bin->right().type() != ASTNode::NODE_NAME
            || !bin->right().cast<ASTName>()->name()->isEqual("__exit__")) {
        return false;
    }

    for (const auto& p : call->pparams()) {
        if (!is_none_like_node(p))
            return false;
    }
    if (!call->kwparams().empty())
        return false;

    if (var_name == NULL)
        return true;
    return bin->left() != NULL
            && bin->left().type() == ASTNode::NODE_NAME
            && bin->left().cast<ASTName>()->name()->isEqual(var_name->value());
}

static bool is_none_exit_call_any_receiver(const PycRef<ASTNode>& node)
{
    return is_exit_call_for_var(node, NULL);
}

static void strip_exit_calls_in_block(const PycRef<ASTBlock>& blk, const PycRef<PycString>& var_name)
{
    if (blk == NULL || var_name == NULL)
        return;
    ASTBlock::list_t filtered;
    for (const auto& child : blk->nodes()) {
        if (is_exit_call_for_var(child, var_name) || is_none_exit_call_any_receiver(child))
            continue;
        PycRef<ASTNodeList> nested_list = child.try_cast<ASTNodeList>();
        if (nested_list != NULL) {
            ASTNodeList::list_t list_filtered;
            for (const auto& ln : nested_list->nodes()) {
                if (is_exit_call_for_var(ln, var_name) || is_none_exit_call_any_receiver(ln))
                    continue;
                PycRef<ASTBlock> nested_blk2 = ln.try_cast<ASTBlock>();
                if (nested_blk2 != NULL)
                    strip_exit_calls_in_block(nested_blk2, var_name);
                list_filtered.emplace_back(ln);
            }
            while (!nested_list->nodes().empty())
                nested_list->removeLast();
            for (const auto& ln : list_filtered)
                nested_list->append(ln);
        } else {
            PycRef<ASTBlock> nested = child.try_cast<ASTBlock>();
            if (nested != NULL)
                strip_exit_calls_in_block(nested, var_name);
        }
        filtered.emplace_back(child);
    }
    while (blk->size() > 0)
        blk->removeLast();
    for (const auto& child : filtered)
        blk->append(child);
}

static ASTBlock::list_t cleanup_linear_statements(const ASTBlock::list_t& lines,
                                                  PycRef<ASTNode>* shared_enter_var = NULL)
{
    ASTBlock::list_t cleaned;
    PycRef<ASTNode> local_enter_var;
    PycRef<ASTNode>* last_enter_var = shared_enter_var != NULL ? shared_enter_var : &local_enter_var;
    for (ASTBlock::list_t::const_iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTNode> node = *it;

        if (node != NULL && node.type() == ASTNode::NODE_NODELIST) {
            node = new ASTNodeList(cleanup_linear_statements(node.cast<ASTNodeList>()->nodes(),
                                                             last_enter_var));
        }
        if (node != NULL && node.type() == ASTNode::NODE_BLOCK) {
            PycRef<ASTBlock> blk = node.cast<ASTBlock>();
            PycRef<ASTNode> saved_enter_var = *last_enter_var;
            std::string saved_enter_sig = cleanup_enter_owner_sig;
            ASTBlock::list_t nested = cleanup_linear_statements(blk->nodes(), last_enter_var);
            *last_enter_var = saved_enter_var;
            cleanup_enter_owner_sig = saved_enter_sig;
            while (blk->size() > 0)
                blk->removeLast();
            for (const auto& ln : nested)
                blk->append(ln);
        }

        if (inModuleCode && node != NULL && node.type() == ASTNode::NODE_RETURN) {
            PycRef<ASTReturn> ret = node.cast<ASTReturn>();
            if (ret->rettype() == ASTReturn::RETURN && is_none_like_node(ret->value()))
                continue;
        }

        if (node != NULL && node.type() == ASTNode::NODE_STORE) {
            PycRef<ASTStore> st = node.cast<ASTStore>();
            if (st->dest() != NULL && st->dest().type() == ASTNode::NODE_NAME
                    && st->src() != NULL && st->src().type() == ASTNode::NODE_CALL) {
                PycRef<ASTCall> enter_call = st->src().cast<ASTCall>();
                if (enter_call->func() != NULL && enter_call->func().type() == ASTNode::NODE_BINARY) {
                    PycRef<ASTBinary> enter_bin = enter_call->func().cast<ASTBinary>();
                    if (enter_bin->op() == ASTBinary::BIN_ATTR
                            && enter_bin->right() != NULL
                            && enter_bin->right().type() == ASTNode::NODE_NAME
                            && enter_bin->right().cast<ASTName>()->name()->isEqual("__enter__")) {
                        *last_enter_var = st->dest();
                        cleanup_enter_owner_sig = node_signature(enter_bin->left());
                    }
                }
            }
        }

        if (*last_enter_var != NULL && node != NULL && node.type() == ASTNode::NODE_CALL) {
            PycRef<ASTCall> call = node.cast<ASTCall>();
            if (call->func() != NULL && call->func().type() == ASTNode::NODE_BINARY) {
                PycRef<ASTBinary> exit_bin = call->func().cast<ASTBinary>();
                if (exit_bin->op() == ASTBinary::BIN_ATTR
                        && exit_bin->right() != NULL
                        && exit_bin->right().type() == ASTNode::NODE_NAME
                        && exit_bin->right().cast<ASTName>()->name()->isEqual("__exit__")) {
                    ASTCall::pparam_t params = normalize_exit_none_params(call->pparams());
                    node = new ASTCall(
                        new ASTBinary(*last_enter_var, new ASTName(exit_bin->right().cast<ASTName>()->name()),
                                      ASTBinary::BIN_ATTR),
                        params, call->kwparams());
                }
            }
        } else if (*last_enter_var != NULL && node != NULL && node.type() == ASTNode::NODE_BINARY) {
            PycRef<ASTBinary> exit_attr = node.cast<ASTBinary>();
            if (exit_attr->op() == ASTBinary::BIN_ATTR
                    && exit_attr->right() != NULL
                    && exit_attr->right().type() == ASTNode::NODE_NAME
                    && exit_attr->right().cast<ASTName>()->name()->isEqual("__exit__")) {
                node = new ASTBinary(*last_enter_var, new ASTName(exit_attr->right().cast<ASTName>()->name()),
                                     ASTBinary::BIN_ATTR);
            }
        }

        if (node != NULL && node.type() == ASTNode::NODE_CALL) {
            PycRef<ASTCall> call = node.cast<ASTCall>();
            if (call->func() != NULL && call->func().type() == ASTNode::NODE_BINARY) {
                PycRef<ASTBinary> bin = call->func().cast<ASTBinary>();
                if (bin->op() == ASTBinary::BIN_ATTR
                        && bin->right() != NULL
                        && bin->right().type() == ASTNode::NODE_NAME
                        && bin->right().cast<ASTName>()->name()->isEqual("__exit__")
                        && call->kwparams().empty()) {
                    ASTCall::pparam_t params = normalize_exit_none_params(call->pparams());
                    if (params.size() != call->pparams().size())
                        node = new ASTCall(call->func(), params, call->kwparams());
                }
            }
        }

        if (node != NULL && node.type() == ASTNode::NODE_CALL && !cleaned.empty()) {
            PycRef<ASTCall> call = node.cast<ASTCall>();
            if (call->func() == NULL && !call->hasVar() && !call->hasKW()
                    && call->kwparams().empty()) {
                bool all_none = true;
                for (ASTCall::pparam_t::const_iterator pit = call->pparams().begin();
                        pit != call->pparams().end(); ++pit) {
                    bool is_none = false;
                    if ((*pit) == NULL) {
                        is_none = true;
                    } else if ((*pit).type() == ASTNode::NODE_OBJECT
                            && (*pit).cast<ASTObject>()->object() == Pyc_None) {
                        is_none = true;
                    } else if ((*pit).type() == ASTNode::NODE_NAME
                            && (*pit).cast<ASTName>()->name()->isEqual("None")) {
                        is_none = true;
                    }
                    if (!is_none) {
                        all_none = false;
                        break;
                    }
                }

                if (all_none) {
                    PycRef<ASTNode> prev = cleaned.back();
                    if (prev != NULL && prev.type() == ASTNode::NODE_BINARY) {
                        PycRef<ASTBinary> bin = prev.cast<ASTBinary>();
                        if (bin->op() == ASTBinary::BIN_ATTR
                                && bin->right() != NULL
                                && bin->right().type() == ASTNode::NODE_NAME
                                && bin->right().cast<ASTName>()->name()->isEqual("__exit__")) {
                            cleaned.back() = new ASTCall(prev, call->pparams(), ASTCall::kwparam_t());
                            continue;
                        }
                    }
                }
            }
        }

        if (node != NULL && node.type() == ASTNode::NODE_KEYWORD
                && node.cast<ASTKeyword>()->key() == ASTKeyword::KW_PASS) {
            ASTBlock::list_t::const_iterator next = it;
            ++next;
            if (next != lines.end() && (*next) != NULL && (*next).type() == ASTNode::NODE_RAISE)
                continue;
        }

        if (!cleaned.empty() && node != NULL && node.type() == ASTNode::NODE_RAISE) {
            PycRef<ASTNode> prev = cleaned.back();
            if (prev != NULL && prev.type() == ASTNode::NODE_RAISE)
                continue;
        }

        if (node != NULL && node.type() == ASTNode::NODE_RETURN && !cleaned.empty()) {
            PycRef<ASTBlock> prev_if = trailing_block_node(cleaned.back());
            if (prev_if != NULL
                    && (prev_if->blktype() == ASTBlock::BLK_IF
                        || prev_if->blktype() == ASTBlock::BLK_ELIF)
                    && prev_if.cast<ASTCondBlock>() != NULL
                    && prev_if.cast<ASTCondBlock>()->negative()
                    && prev_if->size() > 0) {
                bool pass_only = true;
                for (const auto& child : prev_if->nodes()) {
                    if (!is_pass_only_node(child)) {
                        pass_only = false;
                        break;
                    }
                }
                if (pass_only) {
                    while (prev_if->size() > 0)
                        prev_if->removeLast();
                    prev_if->append(node);
                    continue;
                }
            }
        }

        if (!cleaned.empty() && is_terminal_stmt(cleaned.back())) {
            // Drop linear dead code artifacts after a terminal statement.
            continue;
        }

        cleaned.emplace_back(node);
    }
    ASTBlock::list_t patched;
    for (ASTBlock::list_t::const_iterator it = cleaned.begin(); it != cleaned.end(); ++it) {
        PycRef<ASTNode> node = *it;
        PycRef<ASTBlock> forblk = node.try_cast<ASTBlock>();
        if (forblk != NULL && forblk->blktype() == ASTBlock::BLK_FOR) {
            ASTBlock::list_t::const_iterator it_list = it;
            ++it_list;
            ASTBlock::list_t::const_iterator it_store = it_list;
            if (it_list != cleaned.end())
                ++it_store;

            if (it_list != cleaned.end() && it_store != cleaned.end()
                    && is_empty_list_node(*it_list)
                    && (*it_store) != NULL && (*it_store).type() == ASTNode::NODE_STORE) {
                PycRef<ASTStore> store = (*it_store).cast<ASTStore>();
                if (store->dest() != NULL && store->dest().type() == ASTNode::NODE_NAME
                        && store->src() != NULL && store->src().type() == ASTNode::NODE_NAME) {
                    PycRef<PycString> acc_name = store->dest().cast<ASTName>()->name();
                    PycRef<PycString> temp_name = store->src().cast<ASTName>()->name();
                    rewrite_for_append_target(forblk, acc_name);
                    patched.emplace_back(new ASTStore(*it_list, new ASTName(acc_name)));
                    patched.emplace_back(node);

                    it = it_store;
                    ASTBlock::list_t::const_iterator it_restore = it;
                    ++it_restore;
                    if (it_restore != cleaned.end() && (*it_restore) != NULL
                            && (*it_restore).type() == ASTNode::NODE_STORE) {
                        PycRef<ASTStore> restore = (*it_restore).cast<ASTStore>();
                        if (restore->dest() != NULL && restore->dest().type() == ASTNode::NODE_NAME
                                && restore->dest().cast<ASTName>()->name()->isEqual(temp_name->value())
                                && is_none_like_node(restore->src())) {
                            it = it_restore;
                        }
                    }
                    continue;
                }
            }
        }
        patched.emplace_back(node);
    }
    ASTBlock::list_t with_patched;
    for (ASTBlock::list_t::const_iterator it = patched.begin(); it != patched.end(); ++it) {
        PycRef<PycString> with_var;
        PycRef<ASTNode> with_expr;
        if (extract_enter_assignment(*it, with_var, with_expr)) {
            ASTBlock::list_t::const_iterator jt = it;
            ++jt;
            while (jt != patched.end()
                    && !is_exit_call_for_var(*jt, with_var)
                    && !is_none_exit_call_any_receiver(*jt))
                ++jt;
            if (jt != patched.end()) {
                PycRef<ASTWithBlock> withblk = new ASTWithBlock(0);
                withblk->setExpr(with_expr);
                withblk->setVar(new ASTName(with_var));
                withblk->init();
                ASTBlock::list_t::const_iterator body = it;
                ++body;
                for (; body != jt; ++body) {
                    if (is_exit_call_for_var(*body, with_var) || is_none_exit_call_any_receiver(*body))
                        continue;
                    PycRef<ASTBlock> nested = (*body).try_cast<ASTBlock>();
                    if (nested != NULL) {
                        strip_exit_calls_in_block(nested, with_var);
                    } else {
                        PycRef<ASTNodeList> nested_list = (*body).try_cast<ASTNodeList>();
                        if (nested_list != NULL) {
                            ASTNodeList::list_t filtered;
                            for (const auto& ln : nested_list->nodes()) {
                                if (is_exit_call_for_var(ln, with_var) || is_none_exit_call_any_receiver(ln))
                                    continue;
                                PycRef<ASTBlock> ln_blk = ln.try_cast<ASTBlock>();
                                if (ln_blk != NULL)
                                    strip_exit_calls_in_block(ln_blk, with_var);
                                filtered.emplace_back(ln);
                            }
                            while (!nested_list->nodes().empty())
                                nested_list->removeLast();
                            for (const auto& ln : filtered)
                                nested_list->append(ln);
                        }
                    }
                    withblk->append(*body);
                }
                with_patched.emplace_back(withblk.cast<ASTNode>());
                it = jt;
                continue;
            }
        }
        with_patched.emplace_back(*it);
    }

    normalize_nested_with_except(with_patched);
    normalize_sequential_try_regions(with_patched);
    normalize_exception_table_partitions(with_patched);
    normalize_nested_except_handler_partitions(with_patched);
    normalize_trailing_reraise_finally(with_patched);
    return with_patched;
}

static void print_block(PycRef<ASTBlock> blk, PycModule* mod,
                        std::ostream& pyc_output)
{
    ASTBlock::list_t lines = cleanup_linear_statements(
        normalize_except_sequences(blk->nodes(), mod), &cleanup_enter_context);
    ASTBlock::list_t filtered_except_lines;
    for (ASTBlock::list_t::const_iterator it = lines.begin(); it != lines.end(); ++it) {
        PycRef<ASTBlock> exblk = (*it).try_cast<ASTBlock>();
        if (exblk != NULL && exblk->blktype() == ASTBlock::BLK_EXCEPT) {
            ASTBlock::list_t::const_iterator next = it;
            ++next;
            const bool has_later_except = (next != lines.end()
                    && (*next).try_cast<ASTBlock>() != NULL
                    && (*next).try_cast<ASTBlock>()->blktype() == ASTBlock::BLK_EXCEPT);
            if (has_later_except && is_bare_reraise_except_block(exblk))
                continue;
        }
        filtered_except_lines.emplace_back(*it);
    }
    lines.swap(filtered_except_lines);
    if (blk->blktype() == ASTBlock::BLK_WITH) {
        PycRef<ASTNode> with_var_node = blk.cast<ASTWithBlock>()->var();
        PycRef<ASTName> with_var = with_var_node.try_cast<ASTName>();
        if (with_var != NULL) {
            ASTBlock::list_t with_filtered;
            for (const auto& ln : lines) {
                if (is_exit_call_for_var(ln, with_var->name()) || is_none_exit_call_any_receiver(ln))
                    continue;
                PycRef<ASTBlock> nested_blk = ln.try_cast<ASTBlock>();
                if (nested_blk != NULL)
                    strip_exit_calls_in_block(nested_blk, with_var->name());
                with_filtered.emplace_back(ln);
            }
            lines.swap(with_filtered);
        }
    }
    if (blk->blktype() == ASTBlock::BLK_EXCEPT && !lines.empty()) {
        PycRef<PycString> except_alias;
        if (extract_none_init_name(lines.front(), except_alias)) {
            lines.pop_front();
            for (ASTBlock::list_t::iterator it = lines.begin(); it != lines.end();) {
                PycRef<PycString> cleanup_name;
                if (is_delete_of_name(*it, except_alias)
                        || (extract_none_init_name(*it, cleanup_name)
                            && cleanup_name != NULL
                            && cleanup_name->isEqual(except_alias->value())))
                    it = lines.erase(it);
                else
                    ++it;
            }
        }
        strip_trailing_synthetic_loop_keywords(lines);
        strip_trailing_none_cleanup(lines);

        while (lines.size() > 1 && is_pass_only_node(lines.back()))
            lines.pop_back();
    }

    if (lines.size() == 0) {
        PycRef<ASTNode> pass = new ASTKeyword(ASTKeyword::KW_PASS);
        start_line(cur_indent, pyc_output);
        print_src(pass, mod, pyc_output);
    }


    for (auto ln = lines.cbegin(); ln != lines.cend();) {
        if ((*ln).cast<ASTNode>().type() != ASTNode::NODE_NODELIST) {
            start_line(cur_indent, pyc_output);
        }
        print_src(*ln, mod, pyc_output);
        if (++ln != lines.end()) {
            end_line(pyc_output);
        }
    }
}

void print_formatted_value(PycRef<ASTFormattedValue> formatted_value, PycModule* mod,
                           std::ostream& pyc_output)
{
    pyc_output << "{";
    print_src(formatted_value->val(), mod, pyc_output);

    switch (formatted_value->conversion() & ASTFormattedValue::CONVERSION_MASK) {
    case ASTFormattedValue::NONE:
        break;
    case ASTFormattedValue::STR:
        pyc_output << "!s";
        break;
    case ASTFormattedValue::REPR:
        pyc_output << "!r";
        break;
    case ASTFormattedValue::ASCII:
        pyc_output << "!a";
        break;
    }
    if (formatted_value->conversion() & ASTFormattedValue::HAVE_FMT_SPEC) {
        pyc_output << ":" << formatted_value->format_spec().cast<ASTObject>()->object().cast<PycString>()->value();
    }
    pyc_output << "}";
}

static std::unordered_set<ASTNode *> node_seen;

void print_src(PycRef<ASTNode> node, PycModule* mod, std::ostream& pyc_output)
{
    if (node == NULL) {
        pyc_output << "None";
        return;
    }

    if (node_seen.find((ASTNode *)node) != node_seen.end()) {
        fputs("WARNING: Circular reference detected\n", stderr);
        return;
    }
    node_seen.insert((ASTNode *)node);

    switch (node->type()) {
    case ASTNode::NODE_BINARY:
    case ASTNode::NODE_COMPARE:
        {
            PycRef<ASTBinary> bin = node.cast<ASTBinary>();
            print_ordered(node, bin->left(), mod, pyc_output);
            pyc_output << bin->op_str();
            print_ordered(node, bin->right(), mod, pyc_output);
        }
        break;
    case ASTNode::NODE_UNARY:
        {
            PycRef<ASTUnary> un = node.cast<ASTUnary>();
            pyc_output << un->op_str();
            print_ordered(node, un->operand(), mod, pyc_output);
        }
        break;
    case ASTNode::NODE_CALL:
        {
            PycRef<ASTCall> call = node.cast<ASTCall>();
            bool none_like_func = (call->func() == NULL);
            if (!none_like_func && call->func() != NULL && call->func().type() == ASTNode::NODE_NAME) {
                none_like_func = call->func().cast<ASTName>()->name()->isEqual("None");
            } else if (!none_like_func && call->func() != NULL && call->func().type() == ASTNode::NODE_OBJECT) {
                none_like_func = (call->func().cast<ASTObject>()->object() == Pyc_None);
            }
            if (none_like_func && call->kwparams().empty()) {
                bool all_none = true;
                for (const auto& param : call->pparams()) {
                    bool is_none = false;
                    if (param == NULL) {
                        is_none = true;
                    } else if (param.type() == ASTNode::NODE_OBJECT
                            && param.cast<ASTObject>()->object() == Pyc_None) {
                        is_none = true;
                    } else if (param.type() == ASTNode::NODE_NAME
                            && param.cast<ASTName>()->name()->isEqual("None")) {
                        is_none = true;
                    }
                    if (!is_none) {
                        all_none = false;
                        break;
                    }
                }
                if (all_none) {
                    pyc_output << "None";
                    break;
                }
            }
            PycRef<ASTNode> special_call = try_reconstruct_genexpr_call(call->func(), call->pparams(), call->kwparams(), mod);
            if (special_call != NULL) {
                print_src(special_call, mod, pyc_output);
                break;
            }
            print_src(call->func(), mod, pyc_output);
            pyc_output << "(";
            bool first = true;
            for (const auto& param : call->pparams()) {
                if (!first)
                    pyc_output << ", ";
                print_src(param, mod, pyc_output);
                first = false;
            }
            for (const auto& param : call->kwparams()) {
                if (!first)
                    pyc_output << ", ";
                if (param.first.type() == ASTNode::NODE_NAME) {
                    pyc_output << sanitize_identifier(param.first.cast<ASTName>()->name()->value()) << " = ";
                } else {
                    PycRef<PycString> str_name = param.first.cast<ASTObject>()->object().cast<PycString>();
                    pyc_output << sanitize_identifier(str_name->value()) << " = ";
                }
                print_src(param.second, mod, pyc_output);
                first = false;
            }
            if (call->hasVar()) {
                if (!first)
                    pyc_output << ", ";
                pyc_output << "*";
                print_src(call->var(), mod, pyc_output);
                first = false;
            }
            if (call->hasKW()) {
                if (!first)
                    pyc_output << ", ";
                pyc_output << "**";
                print_src(call->kw(), mod, pyc_output);
                first = false;
            }
            pyc_output << ")";
        }
        break;
    case ASTNode::NODE_DELETE:
        {
            pyc_output << "del ";
            print_src(node.cast<ASTDelete>()->value(), mod, pyc_output);
        }
        break;
    case ASTNode::NODE_EXEC:
        {
            PycRef<ASTExec> exec = node.cast<ASTExec>();
            pyc_output << "exec ";
            print_src(exec->statement(), mod, pyc_output);

            if (exec->globals() != NULL) {
                pyc_output << " in ";
                print_src(exec->globals(), mod, pyc_output);

                if (exec->locals() != NULL
                        && exec->globals() != exec->locals()) {
                    pyc_output << ", ";
                    print_src(exec->locals(), mod, pyc_output);
                }
            }
        }
        break;
    case ASTNode::NODE_FORMATTEDVALUE:
        pyc_output << "f" F_STRING_QUOTE;
        print_formatted_value(node.cast<ASTFormattedValue>(), mod, pyc_output);
        pyc_output << F_STRING_QUOTE;
        break;
    case ASTNode::NODE_JOINEDSTR:
        pyc_output << "f" F_STRING_QUOTE;
        for (const auto& val : node.cast<ASTJoinedStr>()->values()) {
            switch (val.type()) {
            case ASTNode::NODE_FORMATTEDVALUE:
                print_formatted_value(val.cast<ASTFormattedValue>(), mod, pyc_output);
                break;
            case ASTNode::NODE_OBJECT:
                // When printing a piece of the f-string, keep the quote style consistent.
                // This avoids problems when ''' or """ is part of the string.
                print_const(pyc_output, val.cast<ASTObject>()->object(), mod, F_STRING_QUOTE);
                break;
            default:
                pyc_output << "{";
                print_src(val, mod, pyc_output);
                pyc_output << "}";
            }
        }
        pyc_output << F_STRING_QUOTE;
        break;
    case ASTNode::NODE_KEYWORD:
        {
            PycRef<ASTKeyword> kw = node.cast<ASTKeyword>();
            if ((kw->key() == ASTKeyword::KW_CONTINUE || kw->key() == ASTKeyword::KW_BREAK)
                    && loop_depth <= 0) {
                pyc_output << "pass";
            } else {
                pyc_output << kw->word_str();
            }
        }
        break;
    case ASTNode::NODE_LIST:
        {
            pyc_output << "[";
            bool first = true;
            cur_indent++;
            for (const auto& val : node.cast<ASTList>()->values()) {
                if (first)
                    pyc_output << "\n";
                else
                    pyc_output << ",\n";
                start_line(cur_indent, pyc_output);
                print_src(val, mod, pyc_output);
                first = false;
            }
            cur_indent--;
            pyc_output << "]";
        }
        break;
    case ASTNode::NODE_SET:
        {
            pyc_output << "{";
            bool first = true;
            cur_indent++;
            for (const auto& val : node.cast<ASTSet>()->values()) {
                if (first)
                    pyc_output << "\n";
                else
                    pyc_output << ",\n";
                start_line(cur_indent, pyc_output);
                print_src(val, mod, pyc_output);
                first = false;
            }
            cur_indent--;
            pyc_output << "}";
        }
        break;
    case ASTNode::NODE_COMPREHENSION:
        {
            PycRef<ASTComprehension> comp = node.cast<ASTComprehension>();

            if (comp->isGenerator())
                pyc_output << "(";
            else
                pyc_output << "[ ";
            print_src(comp->result(), mod, pyc_output);

            for (const auto& gen : comp->generators()) {
                pyc_output << " for ";
                if (gen->index() == NULL)
                    pyc_output << "_";
                else
                    print_src(gen->index(), mod, pyc_output);
                pyc_output << " in ";
                print_src(gen->iter(), mod, pyc_output);
                if (gen->condition()) {
                    pyc_output << " if ";
                    print_src(gen->condition(), mod, pyc_output);
                }
            }
            if (comp->isGenerator())
                pyc_output << ")";
            else
                pyc_output << " ]";
        }
        break;
    case ASTNode::NODE_MAP:
        {
            pyc_output << "{";
            bool first = true;
            cur_indent++;
            for (const auto& val : node.cast<ASTMap>()->values()) {
                if (first)
                    pyc_output << "\n";
                else
                    pyc_output << ",\n";
                start_line(cur_indent, pyc_output);
                print_src(val.first, mod, pyc_output);
                pyc_output << ": ";
                print_src(val.second, mod, pyc_output);
                first = false;
            }
            cur_indent--;
            pyc_output << " }";
        }
        break;
    case ASTNode::NODE_CONST_MAP:
        {
            PycRef<ASTConstMap> const_map = node.cast<ASTConstMap>();
            PycTuple::value_t keys = const_map->keys().cast<ASTObject>()->object().cast<PycTuple>()->values();
            ASTConstMap::values_t values = const_map->values();

            auto map = new ASTMap;
            for (const auto& key : keys) {
                // Values are pushed onto the stack in reverse order.
                PycRef<ASTNode> value = values.back();
                values.pop_back();

                map->add(new ASTObject(key), value);
            }

            print_src(map, mod, pyc_output);
        }
        break;
    case ASTNode::NODE_NAME:
        pyc_output << sanitize_identifier(node.cast<ASTName>()->name()->value());
        break;
    case ASTNode::NODE_NODELIST:
        {
            ASTBlock::list_t lines = cleanup_linear_statements(
                normalize_except_sequences(node.cast<ASTNodeList>()->nodes(), mod),
                &cleanup_enter_context);
            cur_indent++;
            for (const auto& ln : lines) {
                if (ln.cast<ASTNode>().type() != ASTNode::NODE_NODELIST) {
                    start_line(cur_indent, pyc_output);
                }
                print_src(ln, mod, pyc_output);
                end_line(pyc_output);
            }
            cur_indent--;
        }
        break;
    case ASTNode::NODE_BLOCK:
        {
            PycRef<ASTBlock> blk = node.cast<ASTBlock>();
            if (blk->blktype() == ASTBlock::BLK_ELSE && blk->size() == 0)
                break;

            if (blk->blktype() == ASTBlock::BLK_CONTAINER) {
                end_line(pyc_output);
                print_block(blk, mod, pyc_output);
                end_line(pyc_output);
                break;
            }

            PycRef<PycString> except_alias;
            if (blk->blktype() == ASTBlock::BLK_EXCEPT && blk->size() > 0) {
                extract_none_init_name(blk->nodes().front(), except_alias);
            }

            pyc_output << blk->type_str();
            if (blk->blktype() == ASTBlock::BLK_IF
                    || blk->blktype() == ASTBlock::BLK_ELIF
                    || blk->blktype() == ASTBlock::BLK_WHILE) {
                PycRef<ASTCondBlock> condblk = blk.cast<ASTCondBlock>();
                print_condition_expr(condblk->cond(), condblk->negative(), mod, pyc_output, true);
            } else if (blk->blktype() == ASTBlock::BLK_FOR || blk->blktype() == ASTBlock::BLK_ASYNCFOR) {
                pyc_output << " ";
                if (blk.cast<ASTIterBlock>()->index() == NULL)
                    pyc_output << "_";
                else
                    print_src(blk.cast<ASTIterBlock>()->index(), mod, pyc_output);
                pyc_output << " in ";
                print_src(blk.cast<ASTIterBlock>()->iter(), mod, pyc_output);
            } else if (blk->blktype() == ASTBlock::BLK_EXCEPT &&
                    blk.cast<ASTCondBlock>()->cond() != NULL) {
                pyc_output << " ";
                print_src(blk.cast<ASTCondBlock>()->cond(), mod, pyc_output);
                if (except_alias != NULL) {
                    pyc_output << " as ";
                    pyc_output << sanitize_identifier(except_alias->value());
                }
            } else if (blk->blktype() == ASTBlock::BLK_WITH) {
                pyc_output << " ";
                print_src(blk.cast<ASTWithBlock>()->expr(), mod, pyc_output);
                PycRef<ASTNode> var = blk.try_cast<ASTWithBlock>()->var();
                if (var != NULL) {
                    pyc_output << " as ";
                    print_src(var, mod, pyc_output);
                }
            }
            pyc_output << ":\n";

            const bool enters_loop = (blk->blktype() == ASTBlock::BLK_FOR
                    || blk->blktype() == ASTBlock::BLK_WHILE
                    || blk->blktype() == ASTBlock::BLK_ASYNCFOR);
            if (enters_loop)
                ++loop_depth;
            cur_indent++;
            print_block(blk, mod, pyc_output);
            cur_indent--;
            if (enters_loop)
                --loop_depth;
        }
        break;
    case ASTNode::NODE_OBJECT:
        {
            PycRef<PycObject> obj = node.cast<ASTObject>()->object();
            if (obj.type() == PycObject::TYPE_CODE) {
                PycRef<PycCode> code = obj.cast<PycCode>();
                bool savedCleanBuild = cleanBuild;
                int savedLoopDepth = loop_depth;
                loop_depth = 0;
                decompyle(code, mod, pyc_output);
                loop_depth = savedLoopDepth;
                cleanBuild = savedCleanBuild;
            } else {
                print_const(pyc_output, obj, mod);
            }
        }
        break;
    case ASTNode::NODE_PRINT:
        {
            pyc_output << "print ";
            bool first = true;
            if (node.cast<ASTPrint>()->stream() != nullptr) {
                pyc_output << ">>";
                print_src(node.cast<ASTPrint>()->stream(), mod, pyc_output);
                first = false;
            }

            for (const auto& val : node.cast<ASTPrint>()->values()) {
                if (!first)
                    pyc_output << ", ";
                print_src(val, mod, pyc_output);
                first = false;
            }
            if (!node.cast<ASTPrint>()->eol())
                pyc_output << ",";
        }
        break;
    case ASTNode::NODE_RAISE:
        {
            PycRef<ASTRaise> raise = node.cast<ASTRaise>();
            pyc_output << "raise ";
            bool first = true;
            for (const auto& param : raise->params()) {
                if (!first)
                    pyc_output << ", ";
                print_src(param, mod, pyc_output);
                first = false;
            }
        }
        break;
    case ASTNode::NODE_AWAITABLE:
        pyc_output << "await ";
        print_src(node.cast<ASTAwaitable>()->expression(), mod, pyc_output);
        break;
    case ASTNode::NODE_RETURN:
        {
            PycRef<ASTReturn> ret = node.cast<ASTReturn>();
            PycRef<ASTNode> value = ret->value();
            if (!inLambda) {
                switch (ret->rettype()) {
                case ASTReturn::RETURN:
                    pyc_output << "return ";
                    break;
                case ASTReturn::YIELD:
                    pyc_output << "yield ";
                    break;
                case ASTReturn::YIELD_FROM:
                    if (value.type() == ASTNode::NODE_AWAITABLE) {
                        pyc_output << "await ";
                        value = value.cast<ASTAwaitable>()->expression();
                    } else {
                        pyc_output << "yield from ";
                    }
                    break;
                }
            }
            print_src(value, mod, pyc_output);
        }
        break;
    case ASTNode::NODE_SLICE:
        {
            PycRef<ASTSlice> slice = node.cast<ASTSlice>();

            if (slice->op() & ASTSlice::SLICE1) {
                print_src(slice->left(), mod, pyc_output);
            }
            pyc_output << ":";
            if (slice->op() & ASTSlice::SLICE2) {
                print_src(slice->right(), mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_IMPORT:
        {
            PycRef<ASTImport> import = node.cast<ASTImport>();
            if (import->stores().size()) {
                ASTImport::list_t stores = import->stores();

                pyc_output << "from ";
                if (import->name().type() == ASTNode::NODE_IMPORT)
                    print_src(import->name().cast<ASTImport>()->name(), mod, pyc_output);
                else
                    print_src(import->name(), mod, pyc_output);
                pyc_output << " import ";

                if (stores.size() == 1) {
                    auto src = stores.front()->src();
                    auto dest = stores.front()->dest();
                    print_src(src, mod, pyc_output);

                    std::string src_name = src.cast<ASTName>()->name()->value();
                    std::string dest_name = dest.cast<ASTName>()->name()->value();
                    size_t dot = src_name.find('.');
                    bool implicit_pkg_alias = (dot != std::string::npos
                            && src_name.substr(0, dot) == dest_name);
                    if (src_name != dest_name && !implicit_pkg_alias) {
                        pyc_output << " as ";
                        print_src(dest, mod, pyc_output);
                    }
                } else {
                    bool first = true;
                    for (const auto& st : stores) {
                        if (!first)
                            pyc_output << ", ";
                        print_src(st->src(), mod, pyc_output);
                        first = false;

                        std::string src_name = st->src().cast<ASTName>()->name()->value();
                        std::string dest_name = st->dest().cast<ASTName>()->name()->value();
                        size_t dot = src_name.find('.');
                        bool implicit_pkg_alias = (dot != std::string::npos
                                && src_name.substr(0, dot) == dest_name);
                        if (src_name != dest_name && !implicit_pkg_alias) {
                            pyc_output << " as ";
                            print_src(st->dest(), mod, pyc_output);
                        }
                    }
                }
            } else {
                pyc_output << "import ";
                print_src(import->name(), mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_FUNCTION:
        {
            /* Actual named functions are NODE_STORE with a name */
            pyc_output << "(lambda ";
            PycRef<ASTNode> code = node.cast<ASTFunction>()->code();
            PycRef<PycCode> code_src = code.cast<ASTObject>()->object().cast<PycCode>();
            ASTFunction::defarg_t defargs = node.cast<ASTFunction>()->defargs();
            ASTFunction::defarg_t kwdefargs = node.cast<ASTFunction>()->kwdefargs();
            auto da = defargs.cbegin();
            int narg = 0;
            for (int i=0; i<code_src->argCount(); i++) {
                if (narg)
                    pyc_output << ", ";
                pyc_output << sanitize_identifier(code_src->getLocal(narg++)->value());
                if ((code_src->argCount() - i) <= (int)defargs.size()) {
                    pyc_output << " = ";
                    print_src(*da++, mod, pyc_output);
                }
            }
            da = kwdefargs.cbegin();
            if (code_src->kwOnlyArgCount() != 0) {
                pyc_output << (narg == 0 ? "*" : ", *");
                for (int i = 0; i < code_src->argCount(); i++) {
                    pyc_output << ", ";
                    pyc_output << sanitize_identifier(code_src->getLocal(narg++)->value());
                    if ((code_src->kwOnlyArgCount() - i) <= (int)kwdefargs.size()) {
                        pyc_output << " = ";
                        print_src(*da++, mod, pyc_output);
                    }
                }
            }
            pyc_output << ": ";

            if (code_src->name()->isEqual("<lambda>")) {
                inLambda = true;
                print_src(code, mod, pyc_output);
                inLambda = false;
            } else {
                // Anonymous non-lambda code objects (<genexpr>/<listcomp>/...) are
                // expression-only in this context; keep output syntactically valid.
                pyc_output << "None";
            }

            pyc_output << ")";
        }
        break;
    case ASTNode::NODE_STORE:
        {
            PycRef<ASTNode> src = node.cast<ASTStore>()->src();
            PycRef<ASTNode> dest = node.cast<ASTStore>()->dest();
            if (src.type() == ASTNode::NODE_FUNCTION) {
                PycRef<ASTNode> code = src.cast<ASTFunction>()->code();
                PycRef<PycCode> code_src = code.cast<ASTObject>()->object().cast<PycCode>();
                bool isLambda = false;

                if (strcmp(code_src->name()->value(), "<lambda>") == 0) {
                    pyc_output << "\n";
                    start_line(cur_indent, pyc_output);
                    print_src(dest, mod, pyc_output);
                    pyc_output << " = lambda ";
                    isLambda = true;
                } else {
                    pyc_output << "\n";
                    start_line(cur_indent, pyc_output);
                    if (code_src->flags() & PycCode::CO_COROUTINE)
                        pyc_output << "async ";
                    pyc_output << "def ";
                    print_src(dest, mod, pyc_output);
                    pyc_output << "(";
                }

                ASTFunction::defarg_t defargs = src.cast<ASTFunction>()->defargs();
                ASTFunction::defarg_t kwdefargs = src.cast<ASTFunction>()->kwdefargs();
                auto da = defargs.cbegin();
                int narg = 0;
                for (int i = 0; i < code_src->argCount(); ++i) {
                    if (narg)
                        pyc_output << ", ";
                    pyc_output << sanitize_identifier(code_src->getLocal(narg++)->value());
                    if ((code_src->argCount() - i) <= (int)defargs.size()) {
                        pyc_output << " = ";
                        print_src(*da++, mod, pyc_output);
                    }
                }
                da = kwdefargs.cbegin();
                if (code_src->kwOnlyArgCount() != 0) {
                    pyc_output << (narg == 0 ? "*" : ", *");
                    for (int i = 0; i < code_src->kwOnlyArgCount(); ++i) {
                        pyc_output << ", ";
                        pyc_output << sanitize_identifier(code_src->getLocal(narg++)->value());
                        if ((code_src->kwOnlyArgCount() - i) <= (int)kwdefargs.size()) {
                            pyc_output << " = ";
                            print_src(*da++, mod, pyc_output);
                        }
                    }
                }
                if (code_src->flags() & PycCode::CO_VARARGS) {
                    if (narg)
                        pyc_output << ", ";
                    pyc_output << "*" << sanitize_identifier(code_src->getLocal(narg++)->value());
                }
                if (code_src->flags() & PycCode::CO_VARKEYWORDS) {
                    if (narg)
                        pyc_output << ", ";
                    pyc_output << "**" << sanitize_identifier(code_src->getLocal(narg++)->value());
                }

                if (isLambda) {
                    pyc_output << ": ";
                } else {
                    pyc_output << "):\n";
                    printDocstringAndGlobals = true;
                }

                bool preLambda = inLambda;
                inLambda |= isLambda;

                print_src(code, mod, pyc_output);

                inLambda = preLambda;
            } else if (src.type() == ASTNode::NODE_CLASS) {
                pyc_output << "\n";
                start_line(cur_indent, pyc_output);
                pyc_output << "class ";
                print_src(dest, mod, pyc_output);
                PycRef<ASTTuple> bases = src.cast<ASTClass>()->bases().cast<ASTTuple>();
                if (bases->values().size() > 0) {
                    pyc_output << "(";
                    bool first = true;
                    for (const auto& val : bases->values()) {
                        if (!first)
                            pyc_output << ", ";
                        print_src(val, mod, pyc_output);
                        first = false;
                    }
                    pyc_output << "):\n";
                } else {
                    // Don't put parens if there are no base classes
                    pyc_output << ":\n";
                }
                printClassDocstring = true;
                PycRef<ASTNode> code = src.cast<ASTClass>()->code().cast<ASTCall>()
                                       ->func().cast<ASTFunction>()->code();
                print_src(code, mod, pyc_output);
            } else if (src.type() == ASTNode::NODE_IMPORT) {
                PycRef<ASTImport> import = src.cast<ASTImport>();
                if (import->fromlist() != NULL) {
                    PycRef<PycObject> fromlist = import->fromlist().cast<ASTObject>()->object();
                    if (fromlist != Pyc_None) {
                        pyc_output << "from ";
                        if (import->name().type() == ASTNode::NODE_IMPORT)
                            print_src(import->name().cast<ASTImport>()->name(), mod, pyc_output);
                        else
                            print_src(import->name(), mod, pyc_output);
                        pyc_output << " import ";
                        if (fromlist.type() == PycObject::TYPE_TUPLE ||
                                fromlist.type() == PycObject::TYPE_SMALL_TUPLE) {
                            bool first = true;
                            for (const auto& val : fromlist.cast<PycTuple>()->values()) {
                                if (!first)
                                    pyc_output << ", ";
                                pyc_output << val.cast<PycString>()->value();
                                first = false;
                            }
                        } else {
                            pyc_output << fromlist.cast<PycString>()->value();
                        }
                    } else {
                        pyc_output << "import ";
                        print_src(import->name(), mod, pyc_output);
                    }
                } else {
                    pyc_output << "import ";
                    PycRef<ASTNode> import_name = import->name();
                    print_src(import_name, mod, pyc_output);
                    std::string src_name = import_name.cast<ASTName>()->name()->value();
                    std::string dest_name = dest.cast<ASTName>()->name()->value();
                    size_t dot = src_name.find('.');
                    bool implicit_pkg_alias = (dot != std::string::npos
                            && src_name.substr(0, dot) == dest_name);
                    if (src_name != dest_name && !implicit_pkg_alias) {
                        pyc_output << " as ";
                        print_src(dest, mod, pyc_output);
                    }
                }
            } else if (src.type() == ASTNode::NODE_BINARY
                    && src.cast<ASTBinary>()->is_inplace()) {
                print_src(src, mod, pyc_output);
            } else {
                print_src(dest, mod, pyc_output);
                pyc_output << " = ";
                print_src(src, mod, pyc_output);
            }
        }
        break;
    case ASTNode::NODE_CHAINSTORE:
        {
            for (auto& dest : node.cast<ASTChainStore>()->nodes()) {
                print_src(dest, mod, pyc_output);
                pyc_output << " = ";
            }
            print_src(node.cast<ASTChainStore>()->src(), mod, pyc_output);
        }
        break;
    case ASTNode::NODE_SUBSCR:
        {
            print_src(node.cast<ASTSubscr>()->name(), mod, pyc_output);
            pyc_output << "[";
            print_src(node.cast<ASTSubscr>()->key(), mod, pyc_output);
            pyc_output << "]";
        }
        break;
    case ASTNode::NODE_CONVERT:
        {
            pyc_output << "`";
            print_src(node.cast<ASTConvert>()->name(), mod, pyc_output);
            pyc_output << "`";
        }
        break;
    case ASTNode::NODE_TUPLE:
        {
            PycRef<ASTTuple> tuple = node.cast<ASTTuple>();
            ASTTuple::value_t values = tuple->values();
            if (tuple->requireParens())
                pyc_output << "(";
            bool first = true;
            for (const auto& val : values) {
                if (!first)
                    pyc_output << ", ";
                print_src(val, mod, pyc_output);
                first = false;
            }
            if (values.size() == 1)
                pyc_output << ',';
            if (tuple->requireParens())
                pyc_output << ')';
        }
        break;
    case ASTNode::NODE_ANNOTATED_VAR:
        {
            PycRef<ASTAnnotatedVar> annotated_var = node.cast<ASTAnnotatedVar>();
            PycRef<ASTObject> name = annotated_var->name().cast<ASTObject>();
            PycRef<ASTNode> annotation = annotated_var->annotation();

            pyc_output << name->object().cast<PycString>()->value();
            pyc_output << ": ";
            print_src(annotation, mod, pyc_output);
        }
        break;
    case ASTNode::NODE_TERNARY:
        {
            /* parenthesis might be needed
             * 
             * when if-expr is part of numerical expression, ternary has the LOWEST precedence
             *     print(a + b if False else c)
             * output is c, not a+c (a+b is calculated first)
             * 
             * but, let's not add parenthesis - to keep the source as close to original as possible in most cases
             */
            PycRef<ASTTernary> ternary = node.cast<ASTTernary>();
            //pyc_output << "(";
            print_src(ternary->if_expr(), mod, pyc_output);
            const auto if_block = ternary->if_block().cast<ASTCondBlock>();
            pyc_output << " if ";
            print_condition_expr(if_block->cond(), if_block->negative(), mod, pyc_output, false);
            pyc_output << " else ";
            print_src(ternary->else_expr(), mod, pyc_output);
            //pyc_output << ")";
        }
        break;
    case ASTNode::NODE_LOCALS:
        pyc_output << "locals()";
        break;
    case ASTNode::NODE_CLASS:
        {
            PycRef<ASTClass> cls = node.cast<ASTClass>();
            if (cls->name() == NULL) {
                // NULL class name produces 'class None:' which is invalid syntax
                pyc_output << "\n";
                start_line(cur_indent, pyc_output);
                pyc_output << "# AUDIT: class definition could not be decompiled\n";
                start_line(cur_indent, pyc_output);
                pyc_output << "pass";
                cleanBuild = false;
                node_seen.erase((ASTNode *)node);
                return;
            }
            pyc_output << "\n";
            start_line(cur_indent, pyc_output);
            pyc_output << "class ";
            print_src(cls->name(), mod, pyc_output);
            if (cls->bases().type() == ASTNode::NODE_TUPLE) {
                PycRef<ASTTuple> bases = cls->bases().cast<ASTTuple>();
                if (bases->values().size() > 0) {
                    pyc_output << "(";
                    bool first = true;
                    for (const auto& val : bases->values()) {
                        if (!first)
                            pyc_output << ", ";
                        print_src(val, mod, pyc_output);
                        first = false;
                    }
                    pyc_output << ")";
                }
            }
            pyc_output << ":\n";
            printClassDocstring = true;
            /* Safely walk code -> ASTCall -> ASTFunction -> code */
            PycRef<ASTNode> cls_code = cls->code();
            bool class_body_printed = false;
            if (cls_code.type() == ASTNode::NODE_CALL) {
                PycRef<ASTNode> fn = cls_code.cast<ASTCall>()->func();
                if (fn.type() == ASTNode::NODE_FUNCTION) {
                    print_src(fn.cast<ASTFunction>()->code(), mod, pyc_output);
                    class_body_printed = true;
                }
            }
            if (!class_body_printed) {
                start_line(cur_indent + 1, pyc_output);
                pyc_output << "# AUDIT: class body could not be decompiled";
                end_line(pyc_output);
                start_line(cur_indent + 1, pyc_output);
                pyc_output << "pass";
                end_line(pyc_output);
                cleanBuild = false;
            }
        }
        break;
    default:
        pyc_output << "<NODE:" << node->type() << ">";
        fprintf(stderr, "Unsupported Node type: %d\n", node->type());
        cleanBuild = false;
        node_seen.erase((ASTNode *)node);
        return;
    }

    node_seen.erase((ASTNode *)node);
}

bool print_docstring(PycRef<PycObject> obj, int indent, PycModule* mod,
                     std::ostream& pyc_output)
{
    // docstrings are translated from the bytecode __doc__ = 'string' to simply '''string'''
    auto doc = obj.try_cast<PycString>();
    if (doc != nullptr) {
        start_line(indent, pyc_output);
        doc->print(pyc_output, mod, true);
        pyc_output << "\n";
        return true;
    }
    return false;
}

static bool code_has_leading_docstring(PycRef<PycCode> code, PycModule* mod)
{
    if (code == NULL || code->consts() == NULL || code->consts()->size() == 0)
        return false;
    if (code->getConst(0).try_cast<PycString>() == NULL)
        return false;

    PycBuffer source(code->code()->value(), code->code()->length());
    int pos = 0;
    int opcode = Pyc::PYC_INVALID_OPCODE;
    int operand = 0;
    do {
        if (source.atEof())
            return false;
        bc_next(source, mod, opcode, operand, pos);
    } while (opcode == Pyc::CACHE);

    if (opcode != Pyc::LOAD_CONST_A || operand != 0)
        return false;

    int opcode2 = Pyc::PYC_INVALID_OPCODE;
    int operand2 = 0;
    do {
        if (source.atEof())
            return false;
        bc_next(source, mod, opcode2, operand2, pos);
    } while (opcode2 == Pyc::CACHE);

    return opcode2 == Pyc::POP_TOP;
}

static std::unordered_set<PycCode *> code_seen;

void decompyle(PycRef<PycCode> code, PycModule* mod, std::ostream& pyc_output)
{
    if (code_seen.find((PycCode *)code) != code_seen.end()) {
        fputs("WARNING: Circular reference detected\n", stderr);
        return;
    }
    code_seen.insert((PycCode *)code);
    bool savedInModuleCode = inModuleCode;
    inModuleCode = code->name()->isEqual("<module>");

    PycRef<ASTNode> source = BuildFromCode(code, mod);

    PycRef<ASTNodeList> clean = source.cast<ASTNodeList>();
    if (cleanBuild) {
        // The Python compiler adds some stuff that we don't really care
        // about, and would add extra code for re-compilation anyway.
        // We strip these lines out here, and then add a "pass" statement
        // if the cleaned up code is empty
        if (clean->nodes().front().type() == ASTNode::NODE_STORE) {
            PycRef<ASTStore> store = clean->nodes().front().cast<ASTStore>();
            if (store->src().type() == ASTNode::NODE_NAME
                    && store->dest().type() == ASTNode::NODE_NAME) {
                PycRef<ASTName> src = store->src().cast<ASTName>();
                PycRef<ASTName> dest = store->dest().cast<ASTName>();
                if (src->name()->isEqual("__name__")
                        && dest->name()->isEqual("__module__")) {
                    // __module__ = __name__
                    // Automatically added by Python 2.2.1 and later
                    clean->removeFirst();
                }
            }
        }
        if (clean->nodes().front().type() == ASTNode::NODE_STORE) {
            PycRef<ASTStore> store = clean->nodes().front().cast<ASTStore>();
            if (store->src().type() == ASTNode::NODE_OBJECT
                    && store->dest().type() == ASTNode::NODE_NAME) {
                PycRef<ASTObject> src = store->src().cast<ASTObject>();
                PycRef<PycString> srcString = src->object().try_cast<PycString>();
                PycRef<ASTName> dest = store->dest().cast<ASTName>();
                if (dest->name()->isEqual("__qualname__")) {
                    // __qualname__ = '<Class Name>'
                    // Automatically added by Python 3.3 and later
                    clean->removeFirst();
                }
            }
        }

        // Class and module docstrings may only appear at the beginning of their source
        if (printClassDocstring && clean->nodes().front().type() == ASTNode::NODE_STORE) {
            PycRef<ASTStore> store = clean->nodes().front().cast<ASTStore>();
            if (store->dest().type() == ASTNode::NODE_NAME &&
                    store->dest().cast<ASTName>()->name()->isEqual("__doc__") &&
                    store->src().type() == ASTNode::NODE_OBJECT) {
                if (print_docstring(store->src().cast<ASTObject>()->object(),
                        cur_indent + (code->name()->isEqual("<module>") ? 0 : 1), mod, pyc_output))
                    clean->removeFirst();
            }
        }
    }
    // Always strip extraneous trailing return (the compiler always adds one)
    while (clean->nodes().size() > 0 && clean->nodes().back().type() == ASTNode::NODE_RETURN) {
        PycRef<ASTReturn> ret = clean->nodes().back().cast<ASTReturn>();

        if (ret->value() == NULL || ret->value().type() == ASTNode::NODE_LOCALS ||
                is_none_like_node(ret->value())) {
            clean->removeLast();  // Always an extraneous return statement
        } else {
            break;
        }
    }
    if (printClassDocstring)
        printClassDocstring = false;
    // This is outside the clean check so a source block will always
    // be compilable, even if decompylation failed.
    if (clean->nodes().size() == 0 && !code.isIdent(mod->code()))
        clean->append(new ASTKeyword(ASTKeyword::KW_PASS));

    bool part1clean = cleanBuild;
    PycRef<ASTNode> savedCleanupEnterContext = cleanup_enter_context;
    std::string savedCleanupEnterOwnerSig = cleanup_enter_owner_sig;
    PycRef<PycCode> savedCleanupCurrentCode = cleanup_current_code;
    PycModule* savedCleanupCurrentModule = cleanup_current_module;
    cleanup_enter_context = NULL;
    cleanup_enter_owner_sig.clear();
    cleanup_current_code = code;
    cleanup_current_module = mod;

    if (part1clean) {
        if (printDocstringAndGlobals) {
            if (code_has_leading_docstring(code, mod))
                print_docstring(code->getConst(0), cur_indent + 1, mod, pyc_output);

            PycCode::globals_t globs = code->getGlobals();
            if (globs.size()) {
                start_line(cur_indent + 1, pyc_output);
                pyc_output << "global ";
                bool first = true;
                for (const auto& glob : globs) {
                    if (!first)
                        pyc_output << ", ";
                    pyc_output << glob->value();
                    first = false;
                }
                pyc_output << "\n";
            }
            printDocstringAndGlobals = false;
        }

        print_src(source, mod, pyc_output);

        if (!cleanBuild) {
            start_line(cur_indent, pyc_output);
            pyc_output << "# WARNING: Decompyle incomplete\n";
        }
    } else {
        printDocstringAndGlobals = false;

        if (inLambda) {
            pyc_output << "None";
        } else {
            // Emit a syntactically valid fallback body
            int fallback_indent = cur_indent + 1;
            start_line(fallback_indent, pyc_output);
            pyc_output << "# AUDIT: Decompilation incomplete -- AST reconstruction failed\n";
            start_line(fallback_indent, pyc_output);
            pyc_output << "pass\n";
        }
    }

    cleanup_enter_context = savedCleanupEnterContext;
    cleanup_enter_owner_sig = savedCleanupEnterOwnerSig;
    cleanup_current_code = savedCleanupCurrentCode;
    cleanup_current_module = savedCleanupCurrentModule;
    code_seen.erase((PycCode *)code);
    inModuleCode = savedInModuleCode;
}
